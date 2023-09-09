from options import SRBTrajoptOptions
from .srb_builder import SRBBuilder

from functools import partial

from pydrake.all import (
    PiecewisePolynomial,
    MathematicalProgram,
    MultibodyPlant,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    SpatialInertia,
    UnitInertia,
    StartMeshcat,
    MeshcatVisualizer,
    SpatialInertia_,
    RotationalInertia,
    Simulator,
    Context,
    Simulator_,
    RotationMatrix_
)

from pydrake.math import (
    RigidTransform,
    RotationMatrix,
    RotationMatrix_
)
from pydrake.geometry import(
    Rgba,
    Box,
    Sphere
)

from pydrake.autodiffutils import (
    AutoDiffXd,
    ExtractGradient,
    ExtractValue,
    InitializeAutoDiff,
    
)

from pydrake.solvers import (
    SnoptSolver,
    Solve,
    ExpressionConstraint)

from pydrake.symbolic import (
    Expression,
    Variable,
    )
from pydrake.multibody.optimization import (
    QuaternionEulerIntegrationConstraint)

from pydrake.multibody.inverse_kinematics import (
    UnitQuaternionConstraint)

import numpy as np
import time

# for constraint AD evaluations
import jax
import jax.numpy as jnp


# import jax constraints
from .jax_constraints import (
    com_dot_dircol_constraint_jit,
    com_dircol_constraint_jit,
    angvel_dircol_constraint_jit
)
from .jax_utils import (
    np_to_jax,
    batch_np_to_jax,
    quaternion_to_euler,
    euler_to_SO3,
)

from .autodiffxd_utils import (
    autoDiffArrayEqual,
    extract_ad_value_and_gradient
)

import matplotlib.pyplot as plt
class SRBTrajopt:
    def __init__(self, 
                 options: SRBTrajoptOptions,
                 headless: bool = False) -> None:
        self.options = options
        srb_builder = SRBBuilder(self.options, headless)
        self.srb_diagram, self.ad_srb_diagram = srb_builder.create_srb_diagram()
        self.srb_body_idx = self.ad_plant.GetBodyIndices(self.ad_plant.GetModelInstanceByName("body"))[0]
        self.srb_body = self.ad_plant.GetBodyByName("body")
        self.meshcat = srb_builder.meshcat
        self.body_v0 = np.array([0., 0., 0.]) # initial body velocity
        self.ad_simulator = Simulator_[AutoDiffXd](self.ad_srb_diagram, 
                                      self.ad_srb_diagram.CreateDefaultContext())
        self.ad_simulator.Initialize()
        self.I_BBo_B = self.srb_body.default_rotational_inertia().CopyToFullMatrix3()

    @property
    def plant(self):
        """
        Returns:
            plant: MultibodyPlant object from SRB diagram
        """
        return self.srb_diagram.GetSubsystemByName("plant")

    @property
    def ad_plant(self):
        """
        Returns:
            plant: Autodiff MultibodyPlant object from AD SRB diagram
        """
        return self.ad_srb_diagram.GetSubsystemByName("plant")

    @property
    def scene_graph(self):
        """
        Returns:
            scene_graph: SceneGraph object from SRB diagram
        """
        return self.srb_diagram.GetSubsystemByName("scene_graph")

    @property
    def visualizer(self):
        """
        Returns:
            visualizer: MeshcatVisualizer object from SRB diagram
        """
        return self.srb_diagram.GetSubsystemByName("visualizer")

    def create_trajopt_program(self):
        """
        Creates a mathematical program for trajectory optimization, and defines all decision variables
        Returns:
            prog: MathematicalProgram object
        """
        prog = MathematicalProgram()
        self.N = self.options.N
        self.T = self.options.T
        self.mass = self.options.mass
        self.gravity = np.array([0., 0., -9.81])
        ###################################
        # define state decision variables #
        ###################################
        self.h = prog.NewContinuousVariables(self.N - 1, "h")
        self.com = prog.NewContinuousVariables(3, self.N, "com")
        self.com_dot = prog.NewContinuousVariables(3, self.N, "com_dot")
        #self.com_ddot = prog.NewContinuousVariables(3, self.N-1, "com_ddot")

        self.body_quat = prog.NewContinuousVariables(4, self.N, "body_quat")
        self.body_angvel = prog.NewContinuousVariables(3, self.N, "body_angular_vel")
        # position of foot measured from the body frame expressed in world frame
        self.p_W_LF = prog.NewContinuousVariables(3, self.N, "left_foot_pos")
        self.p_W_RF = prog.NewContinuousVariables(3, self.N, "right_foot_pos")

        #####################################
        # define control decision variables #
        #####################################

        self.foot_wrench_names = ["left_front_left", 
                                  "left_front_right", 
                                  "left_back_left", 
                                  "left_back_right",
                                  "right_front_left",
                                  "right_front_right",
                                  "right_back_left",
                                  "right_back_right"]
        foot_names = ["left", "right"]
        contact_forces = [
            prog.NewContinuousVariables(3, self.N - 1, f"foot_{name}_force") for name in self.foot_wrench_names
        ]
        contact_torques = [
            prog.NewContinuousVariables(3, self.N - 1, f"foot_{name}_torque") for name in self.foot_wrench_names
        ]

        self.contact_forces = contact_forces
        self.contact_torques = contact_torques

        return prog
    
    def set_trajopt_initial_guess(self, prog: MathematicalProgram):
        """
        Sets initial guess for SRB state decision variables prior to solver invocation
        
        """
        default_com = np.array([0., 0., 1.])
        default_com_dot = np.array([0., 0., 0.])
        default_quat = np.array([1., 0., 0., 0.])
        default_angvel = np.array([0., 0., 0.])
        default_com_ddot = np.array([0., 0., self.gravity[2]])
        default_p_W_LF = np.array([0., 0.1, -0.6])
        default_p_W_RF = np.array([0., -0.1, -0.6])
        for n in range(self.N):
            # set CoM position guess 
            prog.SetInitialGuess(
                self.com[:, n],
                default_com
            )
            # set CoM velocity guess
            prog.SetInitialGuess(
                self.com_dot[:, n],
                default_com_dot
            )
            # if n < self.N - 1:
            #     # set CoM acceleration guess
            #     prog.SetInitialGuess(
            #         self.com_ddot[:, n],
            #         default_com_ddot
            #     )
            # set body quaternion guess
            prog.SetInitialGuess(
                self.body_quat[:, n],
                default_quat
            )
            # set body angular velocity guess
            prog.SetInitialGuess(
                self.body_angvel[:, n],
                default_angvel
            )
            # set left foot position guess
            prog.SetInitialGuess(
                self.p_W_LF[:, n],
                default_p_W_LF
            )
            # set right foot position guess
            prog.SetInitialGuess(
                self.p_W_RF[:, n],
                default_p_W_RF
            )
            

    #####################################
    # Trajopt cost function definitions #
    #####################################

    def add_com_position_cost(self, prog: MathematicalProgram):
        """
        Adds quadratic error cost on CoM postion decision variables from the reference CoM position
        """
        pass 
    
    def add_foot_position_cost(self, prog: MathematicalProgram):
        """
        Adds quadratic error cost on foot position decision variables from the reference foot positions
        """
        pass 

    def add_control_cost(self, prog: MathematicalProgram):
        """
        Adds quadratic error cost on control decision variables from the reference control values
        """
        pass

    ################################## 
    # Trajopt constraint definitions #
    ##################################
    
    def add_time_scaling_constraint(self, prog: MathematicalProgram):
        """
        Adds bounding box constraints on the duration of h at each knot point
        """
        prog.AddBoundingBoxConstraint(0.5 * self.T / self.N, 2 * self.T / self.N, self.h).evaluator().set_description("duration constraint")
        prog.AddLinearConstraint(sum(self.h) >= 0.9 * self.T).evaluator().set_description("total time constraint lower")
        prog.AddLinearConstraint(sum(self.h) <= 1.1 * self.T).evaluator().set_description("total time constraint upper")

    def add_unit_quaternion_constraint(self, prog: MathematicalProgram):
        """
        Constrains the body quaternion decision variables to be unit quaternions
        """
        for n in range(self.N):
            quat = self.body_quat[:, n]
            prog.AddConstraint(
                UnitQuaternionConstraint(),
                quat
            )

    def add_initial_velocity_constraint(self, prog: MathematicalProgram):
        """
        Constrains the initial CoM velocity to be zero
        """
        
        prog.AddBoundingBoxConstraint(
            self.body_v0, 
            self.body_v0, 
            self.com_dot[:, 0]).evaluator().set_description("initial body velocity constraint")
        
        ### test com velocity constraint
        for n in range(self.N - 1):
            prog.AddBoundingBoxConstraint(
                self.body_v0,
                self.body_v0,
                self.com_dot[:, n]).evaluator().set_description("com velocity constraint")


    def add_quaternion_integration_constraint(self, prog: MathematicalProgram):
        """
        Add euler quaternion integration constraint to ensure that q1 rotates to q2 in dt time for a 
        given angular velocity w assuming that the angular velocity is constant over the time interval
        """
        for n in range(self.N - 1):
            quat1 = self.body_quat[:, n]
            quat2 = self.body_quat[:, n+1]
            angular_vel = self.body_angvel[:, n]
            h = self.h[n]
            # Create the constraint
            dut = QuaternionEulerIntegrationConstraint(allow_quaternion_negation=True)
            # Add the constraint to the program
            prog.AddConstraint(dut, dut.ComposeVariable(quat1, quat2, angular_vel, h))
                           
    def add_com_position_constraint(self, prog: MathematicalProgram):
        """
        Adds direct collocation constraint on CoM positions by integrating CoM velocities
        """
        def com_position_constraint(x: np.ndarray):
            """
            Computes the CoM position constraint from CoM velocity
            """
            h, com_k1, com_dot_k1, foot_forces_k1, com_k2, com_dot_k2 = np.split(x, [
                1,                   # h
                1 + 3,               # com_k1
                1 + 3 + 3,           # com_dot_k1
                1 + 3 + 3 + 24,      # foot forces k1 
                1 + 3 + 3 + 24 + 3   # com_k2
            ])
            foot_forces_k1 = foot_forces_k1.reshape((3, 8), order='F')
            if isinstance(x[0], AutoDiffXd):
                com_k1_val = ExtractValue(com_k1).reshape(3,)
                com_dot_k1_val = ExtractValue(com_dot_k1).reshape(3,)
                foot_forces_k1_val = ExtractValue(foot_forces_k1)
                com_k2_val = ExtractValue(com_k2).reshape(3,)
                com_dot_k2_val = ExtractValue(com_dot_k2).reshape(3,)
                h_val = ExtractValue(h)
                # Compute the constraint value and gradients using JAX
                constraint_val, constraint_jac = com_dircol_constraint_jit(
                    h_val.item(),
                    com_k1_val,
                    com_dot_k1_val,
                    foot_forces_k1_val,
                    com_k2_val,
                    com_dot_k2_val,
                    self.mass,
                    self.gravity)

                constraint_val_ad = InitializeAutoDiff(value=constraint_val,
                                                       gradient=constraint_jac)
                return constraint_val_ad
            else:
                constraint_val, constraint_jac = com_dircol_constraint_jit(
                    h.item(),
                    com_k1,
                    com_dot_k1,
                    foot_forces_k1,
                    com_k2,
                    com_dot_k2,
                    self.mass,
                    self.gravity)
                return np.array(constraint_val, dtype=float).reshape(3,1)
        
        for n in range(self.N - 2):
            # Define a list of variables to concatenate
            comddot_constraint_variables = [
                [self.h[n]],
                self.com[:, n],
                self.com_dot[:, n],
                *[self.contact_forces[i][:, n] for i in range(8)],
                self.com[:, n+1],
                self.com_dot[:, n+1],
                ]
            flattened_variables = [item for sublist in comddot_constraint_variables for item in sublist]
            
            prog.AddConstraint(
                com_position_constraint,
                lb=[0,0,0],
                ub=[0,0,0],
                vars=np.array(flattened_variables),
                description=f"com_position_constraint_{n}"
            )
    
    def add_com_velocity_constraint(self, prog: MathematicalProgram):
        """
        Adds direct collocation constraint on CoM velocities by evaluating the dynamics x_ddot = 1/m * sum(F_i) + g
        """
        def com_velocity_constraint(x: np.ndarray):
            """
            Computes the CoM velocity constraint for a given CoM velocity and contact forces
            com_ddot = 1/m * sum(F_i) + g
            """
            h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2 = np.split(x, [
                1,              # h
                1 + 3,          # com_dot_k1
                1 + 3 + 24,     # foot_forces_k1
                1 + 3 + 24 + 3  # com_dot_k2
            ])
            foot_forces_k1 = foot_forces_k1.reshape((3, 8), order='F')
            foot_forces_k2 = foot_forces_k2.reshape((3, 8), order='F')
            # Check if the first element of x is an AutoDiffXd object
            if isinstance(x[0], AutoDiffXd):
                com_dot_k1_val = ExtractValue(com_dot_k1).reshape(3,)
                foot_forces_k1_val = ExtractValue(foot_forces_k1)
                com_dot_k2_val = ExtractValue(com_dot_k2).reshape(3,)
                foot_forces_k2_val = ExtractValue(foot_forces_k2)
                h_val = ExtractValue(h)
                # Compute the constraint value and gradients using JAX
                constraint_val, constraint_jac = com_dot_dircol_constraint_jit(
                    h_val.item(),
                    com_dot_k1_val,
                    foot_forces_k1_val,
                    com_dot_k2_val,
                    foot_forces_k2_val,
                    self.mass,
                    self.gravity)
                constraint_val_ad = InitializeAutoDiff(value=constraint_val,
                                                       gradient=constraint_jac)
                return constraint_val_ad
            else:

                constraint_val, constraint_jac = com_dot_dircol_constraint_jit(
                    h.item(),
                    com_dot_k1,
                    foot_forces_k1,
                    com_dot_k2,
                    foot_forces_k2,
                    self.mass,
                    self.gravity)
                return np.array(constraint_val, dtype=float).reshape(3,1)

        for n in range(self.N - 2):
            # Define a list of variables to concatenate
            comddot_constraint_variables = [
                [self.h[n]],
                self.com_dot[:, n],
                *[self.contact_forces[i][:, n] for i in range(8)],
                self.com_dot[:, n+1],
                *[self.contact_forces[i][:, n+1] for i in range(8)],]
            flattened_variables = [item for sublist in comddot_constraint_variables for item in sublist]
            
            prog.AddConstraint(
                com_velocity_constraint,
                lb=[0,0,0],
                ub=[0,0,0],
                vars=np.array(flattened_variables),
                description=f"com_velocity_constraint_{n}"
            )

    def add_angular_velocity_constraint(self, prog: MathematicalProgram):
        """
        Integrates the contact torque decision variables to constrain angular velocity decision variables
        """

        # Create autodiff contexts for this constraint (to maximize cache hits)
        ad_angular_velocity_dynamics_context = [
            self.ad_plant.CreateDefaultContext() for i in range(self.N)
        ]

        def angular_velocity_constraint(x: np.ndarray, context_index: int):
            # Define variable names and their corresponding indices
            # Split the input array using the indices
            split_vars = np.split(x, [
                1,                   # h
                4,                   # com_k1
                8,                   # quat_k1
                11,                  # com_dot_k1
                14,                  # body_angvel_k1
                17,                  # p_B_LF_W_k1
                20,                  # p_B_RF_W_k1
                44,                  # foot_forces_k1
                68,                  # foot_torques_k1
                71,                  # com_k2
                75,                  # quat_k2
                78,                  # com_dot_k2
                81,                  # body_angvel_k2
                84,                  # p_B_LF_W_k2
                87,                  # p_B_RF_W_k2
                111,                 # foot_forces_k2
            ])
            # Unpack the split variables into meaningful names
            h, com_k1, quat_k1, com_dot_k1, body_angvel_k1, p_W_LF_k1, p_W_RF_k1, \
            foot_forces_k1, foot_torques_k1, com_k2, quat_k2, com_dot_k2, body_angvel_k2, \
            p_W_LF_k2, p_W_RF_k2, foot_forces_k2, foot_torques_k2 = split_vars

            foot_forces_k1 = foot_forces_k1.reshape((3, len(self.foot_wrench_names)), order='F')
            foot_forces_k2 = foot_forces_k1.reshape((3, len(self.foot_wrench_names)), order='F')

            foot_torques_k1 = foot_torques_k1.reshape((3, len(self.foot_wrench_names)), order='F')
            foot_torques_k2 = foot_torques_k2.reshape((3, len(self.foot_wrench_names)), order='F')

            k1_decision_vars = {}
            k2_decision_vars = {}

            srb_pos_k1 = np.concatenate((quat_k1, com_k1))
            srb_pos_k2 = np.concatenate((quat_k2, com_k2))
            if not autoDiffArrayEqual(srb_pos_k1,
                                      self.ad_plant.GetPositions(
                                          ad_angular_velocity_dynamics_context[context_index])):
                self.ad_plant.SetPositions(
                    ad_angular_velocity_dynamics_context[context_index], 
                    srb_pos_k1)
            
            if not autoDiffArrayEqual(srb_pos_k2,
                                      self.ad_plant.GetPositions(
                                          ad_angular_velocity_dynamics_context[context_index + 1])):
                
                self.ad_plant.SetPositions(
                    ad_angular_velocity_dynamics_context[context_index + 1],
                    srb_pos_k2)
            
            k1_decision_vars["com_k1"] = ExtractValue(com_k1).reshape(3,)
            k1_decision_vars["quat_k1"] = ExtractValue(quat_k1).reshape(4,)
            k1_decision_vars["com_dot_k1"] = ExtractValue(com_dot_k1).reshape(3,)
            k1_decision_vars["body_angvel_k1"] = ExtractValue(body_angvel_k1).reshape(3,)
            k1_decision_vars["p_W_LF_k1"] = ExtractValue(p_W_LF_k1).reshape(3,)
            k1_decision_vars["p_W_RF_k1"] = ExtractValue(p_W_RF_k1).reshape(3,)
            k1_decision_vars["foot_forces_k1"] = ExtractValue(foot_forces_k1)
            k1_decision_vars["foot_torques_k1"] = ExtractValue(foot_torques_k1)

            k2_decision_vars["com_k2"] = ExtractValue(com_k2).reshape(3,)
            k2_decision_vars["quat_k2"] = ExtractValue(quat_k2).reshape(4,)
            k2_decision_vars["com_dot_k2"] = ExtractValue(com_dot_k2).reshape(3,)
            k2_decision_vars["body_angvel_k2"] = ExtractValue(body_angvel_k2).reshape(3,)
            k2_decision_vars["p_W_LF_k2"] = ExtractValue(p_W_LF_k2).reshape(3,)
            k2_decision_vars["p_W_RF_k2"] = ExtractValue(p_W_RF_k2).reshape(3,)
            k2_decision_vars["foot_forces_k2"] = ExtractValue(foot_forces_k2)
            k2_decision_vars["foot_torques_k2"] = ExtractValue(foot_torques_k2)
            
            if isinstance(x[0], AutoDiffXd):
                h_val = ExtractValue(h)
                constraint_val, constraint_jac = angvel_dircol_constraint_jit(
                    h_val.item(),
                    k1_decision_vars,
                    k2_decision_vars,
                    self.options.foot_length,
                    self.options.foot_width,
                    self.I_BBo_B,
                    self.mass
                )

                constraint_val_ad = InitializeAutoDiff(value=constraint_val,
                                                       gradient=constraint_jac)
                return constraint_val_ad
            else:   
                constraint_val, constraint_jac = angvel_dircol_constraint_jit(
                    h.item(),
                    k1_decision_vars,
                    k2_decision_vars,
                    self.options.foot_length,
                    self.options.foot_width,
                    self.I_BBo_B,
                    self.mass
                )
                return np.array(constraint_val, dtype=float).reshape(3,1)

        for n in range(self.N - 2):
            # Define a list of variables to concatenate
            comddot_constraint_variables = [
                [self.h[n]],
                self.com[:, n],
                self.body_quat[:, n],
                self.com_dot[:, n],
                self.body_angvel[:, n],
                self.p_W_LF[:, n],
                self.p_W_RF[:, n],
                *[self.contact_forces[i][:, n] for i in range(8)],
                *[self.contact_torques[i][:, n] for i in range(8)],
                self.com[:, n + 1],
                self.body_quat[:, n + 1],
                self.com_dot[:, n + 1],
                self.body_angvel[:, n + 1],
                self.p_W_LF[:, n + 1],
                self.p_W_RF[:, n + 1],
                *[self.contact_forces[i][:, n + 1] for i in range(8)],
                *[self.contact_torques[i][:, n + 1] for i in range(8)],]
            
            flattened_variables = [item for sublist in comddot_constraint_variables for item in sublist]

            prog.AddConstraint(
                partial(angular_velocity_constraint, context_index=n),
                lb=[0,0,0],
                ub=[0,0,0],
                vars=np.array(flattened_variables),
                description=f"com_velocity_constraint_{n}"
            )

    def add_linear_absolute_value_constraint(self, 
                                             prog: MathematicalProgram,
                                             var,
                                             lim):
        """
        Adds linear absolute value constraint on a decision variable
        """
        prog.AddLinearConstraint(
            var <= lim
        )
        prog.AddLinearConstraint(
            -var <= lim
        )

    def add_contact_wrench_cone_constraint(self, prog: MathematicalProgram):
        """
        Constrains the contact wrench decision variables to lie within the wrench cone set 
        
        """
        mu = self.options.mu
        mu = 0.9  
        max_z_grf = float(self.options.max_z_grf)
        foot_half_l = float(self.options.foot_length/2)
        foot_half_w = float(self.options.foot_width/2)
        for n in range(self.N - 1):
            left_foot_x_frc_sum = sum(self.contact_forces[i][0, n] for i in range(0, 4))
            left_foot_y_frc_sum = sum(self.contact_forces[i][1, n] for i in range(0, 4))
            left_foot_z_frc_sum = sum(self.contact_forces[i][2, n] for i in range(0, 4))

            left_foot_y_toe_frc_sum = sum(self.contact_forces[i][1, n] for i in range(0, 2))
            left_foot_y_heel_frc_sum = sum(self.contact_forces[i][1, n] for i in range(2, 4))
            left_foot_z_toe_frc_sum = sum(self.contact_forces[i][2, n] for i in range(0, 2))
            left_foot_z_heel_frc_sum = sum(self.contact_forces[i][2, n] for i in range(2, 4))
            left_foot_z_lhs_frc_sum = sum(self.contact_forces[i][2, n] for i in [0, 2])
            left_foot_z_rhs_frc_sum = sum(self.contact_forces[i][2, n] for i in [1, 3])

            right_foot_x_frc_sum = sum(self.contact_forces[i][0, n] for i in range(4, 8))
            right_foot_y_frc_sum = sum(self.contact_forces[i][1, n] for i in range(4, 8))
            right_foot_z_frc_sum = sum(self.contact_forces[i][2, n] for i in range(4, 8))

            right_foot_y_toe_frc_sum = sum(self.contact_forces[i][1, n] for i in range(4, 6))
            right_foot_y_heel_frc_sum = sum(self.contact_forces[i][1, n] for i in range(6, 8))
            right_foot_z_toe_frc_sum = sum(self.contact_forces[i][2, n] for i in range(4, 6))
            right_foot_z_heel_frc_sum = sum(self.contact_forces[i][2, n] for i in range(6, 8))
            right_foot_z_lhs_frc_sum = sum(self.contact_forces[i][2, n] for i in [4, 6])
            right_foot_z_rhs_frc_sum = sum(self.contact_forces[i][2, n] for i in [5, 7])

            left_foot_m_x_sum = sum(self.contact_torques[i][0, n] for i in range(0, 4))
            left_foot_m_y_sum = sum(self.contact_torques[i][1, n] for i in range(0, 4))
            left_foot_m_z_sum = sum(self.contact_torques[i][2, n] for i in range(0, 4))

            right_foot_m_x_sum = sum(self.contact_torques[i][0, n] for i in range(4, 8))
            right_foot_m_y_sum = sum(self.contact_torques[i][1, n] for i in range(4, 8))
            right_foot_m_z_sum = sum(self.contact_torques[i][2, n] for i in range(4, 8)) 

            #Max Z GRF constraints 
            self.add_linear_absolute_value_constraint(
                prog,
                left_foot_z_frc_sum,
                max_z_grf
            )
            self.add_linear_absolute_value_constraint(
                prog,
                right_foot_z_frc_sum,
                max_z_grf
            )

            for i in range(len(self.contact_forces)):
                prog.AddBoundingBoxConstraint(
                    0.,
                    np.inf,
                    self.contact_forces[i][2, n],
                    ).evaluator().set_description("nonnegative foot z force")

            # Linear friction cone constraints
            self.add_linear_absolute_value_constraint(
                prog,
                left_foot_x_frc_sum,
                mu*left_foot_z_frc_sum
            )
            self.add_linear_absolute_value_constraint(
                prog,
                left_foot_y_frc_sum,
                mu*left_foot_z_frc_sum
            )
            self.add_linear_absolute_value_constraint(
                prog,
                right_foot_y_frc_sum,
                mu*right_foot_z_frc_sum
            )
            self.add_linear_absolute_value_constraint(
                prog,
                right_foot_x_frc_sum,
                mu*right_foot_z_frc_sum
            )

            # Contact Torque Constraints
            # The origin is assumed to be at the center of the 
            # foot polygon

            # M_y max torque constraints
            prog.AddLinearConstraint(
                -left_foot_z_frc_sum*foot_half_l <= \
                left_foot_m_y_sum
            )
            prog.AddLinearConstraint(
                left_foot_m_y_sum <= \
                left_foot_z_frc_sum*foot_half_l
            )
            prog.AddLinearConstraint(
                -right_foot_z_frc_sum*foot_half_l <= \
                right_foot_m_y_sum
            )
            prog.AddLinearConstraint(
                right_foot_m_y_sum <= \
                right_foot_z_frc_sum*foot_half_l
            )

            # M_y torque constraints
            prog.AddLinearConstraint(
                left_foot_m_y_sum == \
                    left_foot_z_heel_frc_sum*foot_half_l - \
                    left_foot_z_toe_frc_sum*foot_half_l
            )

            prog.AddLinearConstraint(
                right_foot_m_y_sum == \
                    right_foot_z_heel_frc_sum*foot_half_l - \
                    right_foot_z_toe_frc_sum*foot_half_l
            )
            #############################################


            # M_x max torque constraints
            prog.AddLinearConstraint(
                -left_foot_z_frc_sum*foot_half_w <= \
                left_foot_m_x_sum
            )
            prog.AddLinearConstraint(
                left_foot_m_x_sum <= \
                left_foot_z_frc_sum*foot_half_w
            )
            prog.AddLinearConstraint(
                -right_foot_z_frc_sum*foot_half_w <= \
                right_foot_m_x_sum
            )
            prog.AddLinearConstraint(
                right_foot_m_x_sum <= \
                right_foot_z_frc_sum*foot_half_w
            )

            prog.AddLinearConstraint(
                left_foot_m_x_sum == \
                    left_foot_z_lhs_frc_sum*foot_half_w - \
                    left_foot_z_rhs_frc_sum*foot_half_w
            )
            prog.AddLinearConstraint(
                right_foot_m_x_sum == \
                    right_foot_z_lhs_frc_sum*foot_half_w - \
                    right_foot_z_rhs_frc_sum*foot_half_w
            )

            # Mz max torque constraints

            # Friction cone constraints for left foot heel and toe contacts
            self.add_linear_absolute_value_constraint(
                prog,
                left_foot_y_toe_frc_sum,
                mu*left_foot_z_toe_frc_sum
            )
            self.add_linear_absolute_value_constraint(
                prog,
                left_foot_y_heel_frc_sum,
                mu*left_foot_z_heel_frc_sum
            )
            self.add_linear_absolute_value_constraint(
                prog,
                right_foot_y_toe_frc_sum,
                mu*right_foot_z_toe_frc_sum
            )
            self.add_linear_absolute_value_constraint(
                prog,
                right_foot_y_heel_frc_sum,
                mu*right_foot_z_heel_frc_sum
            )

            prog.AddLinearConstraint(
                left_foot_m_z_sum == \
                    left_foot_y_toe_frc_sum*foot_half_l - \
                    left_foot_y_heel_frc_sum*foot_half_l
            )

            prog.AddLinearConstraint(
                right_foot_m_z_sum == \
                    right_foot_y_toe_frc_sum*foot_half_l - \
                    right_foot_y_heel_frc_sum*foot_half_l
            )

    def add_step_length_kinematic_constraint(self, prog: MathematicalProgram):
        """
        Constrains the step location and leg extension to be within kinematically feasible bounds
        """
        for n in range(self.N):
            # implements the linear constraint |p_B_LF_W| <= lim
            for i in range(3):
                p_B_LF_W_i = self.p_W_LF[i, n] - self.com[i, n]
                p_B_RF_W_i = self.p_W_RF[i, n] - self.com[i, n]

                prog.AddLinearConstraint(
                    v= [p_B_LF_W_i],
                    lb= [-np.inf],
                    ub= [self.options.leg_extension_bounds[i]]
                )
                prog.AddLinearConstraint(
                    v= [-p_B_LF_W_i],
                    lb= [-np.inf],
                    ub= [self.options.leg_extension_bounds[i]]
                )
                prog.AddLinearConstraint(
                    v= [p_B_RF_W_i],
                    lb= [-np.inf],
                    ub= [self.options.leg_extension_bounds[i]]
                )
                prog.AddLinearConstraint(
                    v= [-p_B_RF_W_i],
                    lb= [-np.inf],
                    ub= [self.options.leg_extension_bounds[i]]
                )

    def add_foot_velocity_kinematic_constraint(self, prog: MathematicalProgram):
        """
        Constrains the stance foot velocity to be zero
        """

        def foot_velocity_constraint(x: np.ndarray,
                                     i: int,):
            p_W_F_k1, p_W_F_k2 = np.split(x, [
                3,
            ])

            # Check if the first element of x is an AutoDiffXd object
            # if isinstance(x[0], AutoDiffXd):
            p_W_F_con_k1 = p_W_F_k1[0:3].copy()
            p_W_F_con_k2 = p_W_F_k2[0:3].copy()
            if i in [0, 4]:
                p_W_F_con_k1[0] += self.options.foot_length/2
                p_W_F_con_k1[1] += self.options.foot_width/2
                p_W_F_con_k2[0] += self.options.foot_length/2
                p_W_F_con_k2[1] += self.options.foot_width/2
            elif i in [1, 5]:
                p_W_F_con_k1[0] += self.options.foot_length/2
                p_W_F_con_k1[1] -= self.options.foot_width/2
                p_W_F_con_k2[0] += self.options.foot_length/2
                p_W_F_con_k2[1] -= self.options.foot_width/2
            elif i in [2, 6]:
                p_W_F_con_k1[0] -= self.options.foot_length/2
                p_W_F_con_k1[1] += self.options.foot_width/2
                p_W_F_con_k2[0] -= self.options.foot_length/2
                p_W_F_con_k2[1] += self.options.foot_width/2
            elif i in [3, 7]:
                p_W_F_con_k1[0] -= self.options.foot_length/2
                p_W_F_con_k1[1] -= self.options.foot_width/2
                p_W_F_con_k2[0] -= self.options.foot_length/2
                p_W_F_con_k2[1] -= self.options.foot_width/2
            return p_W_F_con_k2 - p_W_F_con_k1

        for n in range(1, self.N):
            for i in range(len(self.contact_forces)):
                if self.in_stance[i, n - 1]:
                    if i in range(4):
                        constraint_args = np.concatenate([
                            self.p_W_LF[:, n],
                            self.p_W_LF[:, n-1],
                            ])
                        prog.AddConstraint(
                            partial(foot_velocity_constraint, i = i),
                            lb=[0., 0., 0.],
                            ub=[0., 0., 0.],
                            vars=constraint_args,
                            description=f"foot_{i}_velocity_constraint_{n}"
                        )
                        prog.AddLinearConstraint(
                            self.p_W_LF[2,n] == 0
                        ).evaluator().set_description(f"foot_{i}_stance_z_pos_constraint_{n}")

                    elif i in range(4, 8):
                        constraint_args = np.concatenate([
                            self.p_W_RF[:, n],
                            self.p_W_RF[:, n-1],
                            ])
                        prog.AddConstraint(
                            partial(foot_velocity_constraint, i = i),
                            lb=[0., 0., 0.],
                            ub=[0., 0., 0.],
                            vars=constraint_args,
                            description=f"foot_{i}_velocity_constraint_{n}"
                        )
                        prog.AddLinearConstraint(
                            self.p_W_RF[2,n] == 0
                        ).evaluator().set_description(f"foot_{i}_stace_z_pos_constraint_{n}")
                else:
                    if i in range(4):
                        prog.AddLinearConstraint(
                            self.p_W_LF[2,n] >= 0.001
                        ).evaluator().set_description(f"foot_{i}_flight_z_pos_constraint_{n}")
                    elif i in range(4, 8):
                        prog.AddLinearConstraint(
                            self.p_W_RF[2,n] >= 0.001
                        ).evaluator().set_description(f"foot_{i}_flight_z_pos_constraint_{n}")

    def formulate_trajopt_problem(self,):
        """
        Formulates the trajectory optimization problem by defining all costs and constraints
        """
        prog = self.create_trajopt_program()
        self.create_contact_sequence()
        self.set_trajopt_initial_guess(prog)
        self.add_time_scaling_constraint(prog)
        self.add_unit_quaternion_constraint(prog)
        self.add_quaternion_integration_constraint(prog)
        self.add_com_velocity_constraint(prog)
        self.add_com_position_constraint(prog)
        self.add_initial_velocity_constraint(prog)
        self.add_step_length_kinematic_constraint(prog)
        self.add_angular_velocity_constraint(prog)
        self.add_contact_wrench_cone_constraint(prog)
        self.add_foot_velocity_kinematic_constraint(prog)

        return prog

    def configure_snopt_solver(self, prog: MathematicalProgram):
        snopt = SnoptSolver().solver_id()
        prog.SetSolverOption(
            snopt, "Iterations Limits", 1e5
        )
        prog.SetSolverOption(
            snopt, "Major Iterations Limit", 200 
        )
        prog.SetSolverOption(snopt, "Major Feasibility Tolerance", 5e-6)
        prog.SetSolverOption(snopt, "Major Optimality Tolerance", 1e-3)
        prog.SetSolverOption(snopt, "Superbasics limit", 2000)
        prog.SetSolverOption(snopt, "Major print level", 11)

        prog.SetSolverOption(snopt, "Linesearch tolerance", 0.99)
        prog.SetSolverOption(snopt, 'Print file', 'snopt.out')

    def solve_trajopt(self,):
        """
        Solve the completed optimization problem with all constraints and costs defined
        """
        prog = self.formulate_trajopt_problem()
        snopt = SnoptSolver()
        res = snopt.Solve(prog)
        print(res.is_success())  # We expect this to be false if iterations are limited.
        print(res.get_solver_details().info)
        print(f"solution cost: {res.get_optimal_cost()}")
        # solve again with collision constraints
        # set initial guess
        if not res.is_success():
            constraints = res.GetInfeasibleConstraintNames(prog,)
            print("Infeasible constraints:")
            for c in constraints:
                print(c)
        # print(res.GetSolution(self.h))
        # print(np.sum(res.GetSolution(self.h)))
        # print(res.GetSolution(self.body_quat))
        #print(res.GetSolution(self.com))
        # plot result 
        left_foot_z_frc_sum = []
        right_foot_z_frc_sum = []
        # for i in range(4):
        #     contact_frc_soln = res.GetSolution(self.contact_forces[i])
        for i in range(self.N-1):
            contact_frc_sum = 0
            for j in range(4):
                contact_frc_soln = res.GetSolution(self.contact_forces[j])
                contact_frc_sum += contact_frc_soln[2, i]
            left_foot_z_frc_sum.append(contact_frc_sum)
            
        for i in range(self.N-1):
            contact_frc_sum = 0
            for j in range(4, 8):
                contact_frc_soln = res.GetSolution(self.contact_forces[j])
                contact_frc_sum += contact_frc_soln[2, i]
            right_foot_z_frc_sum.append(contact_frc_sum)
        
        time = np.cumsum(np.hstack((0, res.GetSolution(self.h))))
        plt.plot(time, res.GetSolution(self.com)[2, :])
        plt.show()
        plt.plot(time[:-1], left_foot_z_frc_sum)
        plt.plot(time[:-1], right_foot_z_frc_sum)
        plt.show()
        # print(res.GetSolution(self.contact_forces[0]))
        # print(res.GetSolution(self.com_dot))

        return res

    def create_contact_sequence(self):

        N = self.options.N
        # use placeholder walking gait contact sequnce for now 
        in_stance = np.ones((8, N))
        # in_stance[0:4, 0:int(N/2)] = 1
        # in_stance[4:8, int(N/2):N] = 1
        self.in_stance = in_stance

    def render_srb(self) -> None:
        """
        Visualizes the SRB in Drake Visualizer
        """
        self.meshcat.Delete()
        diagram_context = self.srb_diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyContextFromRoot(diagram_context)
        X_WB = RigidTransform()
        X_WB.set_translation([0, 0, 1])
        self.plant.SetFreeBodyPose(plant_context, self.plant.GetBodyByName("body"), X_WB)
        # render the plant in Drake Visualizer
        self.srb_diagram.ForcedPublish(diagram_context)
        time.sleep(3.0)




