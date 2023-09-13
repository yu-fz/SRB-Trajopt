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
    RotationMatrix_,
    PiecewiseQuaternionSlerp,
    PiecewiseQuaternionSlerp_,
    Quaternion,
    Quaternion_
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
    ExpressionConstraint,
    QuadraticCost)

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
    com_dot_dircol_constraint_jax,
    com_dircol_constraint_jax,
    angvel_dircol_constraint_jax,
    compute_omega_dot_jax
)
from .jax_utils import (
    np_to_jax,
    batch_np_to_jax,
    quaternion_to_euler,
    euler_to_SO3,
)

from .autodiffxd_utils import (
    autodiff_array_equal,
    extract_ad_value_and_gradient,
    multiply_and_sum
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
        self.ad_srb_body = self.ad_plant.GetBodyByName("body")
        self.srb_body = self.plant.GetBodyByName("body")
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
        self.mg = -self.mass * self.gravity[2]
        ###################################
        # define state decision variables #
        ###################################
        self.h = prog.NewContinuousVariables(self.N - 1, "h")
        self.com = prog.NewContinuousVariables(3, self.N, "com")
        self.com_dot = prog.NewContinuousVariables(3, self.N, "com_dot")

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
            prog.NewContinuousVariables(3, self.N, f"foot_{name}_force") for name in self.foot_wrench_names
        ]
        contact_torques = [
            prog.NewContinuousVariables(3, self.N, f"foot_{name}_torque") for name in self.foot_wrench_names
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
        #default_angvel = np.array([0., 0., 0.])
        default_p_W_LF = np.array([0., 0.1, 0.])
        default_p_W_RF = np.array([0., -0.1, 0.])
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
            # set body quaternion guess
            prog.SetInitialGuess(
                self.body_quat[:, n],
                default_quat
            )
        
        for n in range(self.N):
            for i in range(len(self.contact_forces)):
                if self.in_stance[i, n]:
                    # set foot force guess
                    prog.SetInitialGuess(
                        self.contact_forces[i][2, n],
                        self.mg/len(self.contact_forces)
                    )
                else:
                    # set foot force guess
                    prog.SetInitialGuess(
                        self.contact_forces[i][2, n],
                        0
                    )


            # # set body angular velocity guess
            # prog.SetInitialGuess(
            #     self.body_angvel[:, n],
            #     default_angvel
            # )
            # # set left foot position guess
            # prog.SetInitialGuess(
            #     self.p_W_LF[:, n],
            #     default_p_W_LF
            # )
            # # set right foot position guess
            # prog.SetInitialGuess(
            #     self.p_W_RF[:, n],
            #     default_p_W_RF
            # )
            

    #####################################
    # Trajopt cost function definitions #
    #####################################

    def add_com_position_cost(self, prog: MathematicalProgram):
        """
        Adds quadratic error cost on CoM postion decision variables from the reference CoM position
        """
        Q = np.eye(3)
        Q[0, 0] = 0.001
        Q[1, 1] = 0.001
        Q[2, 2] = 0.005
        x_des = np.array([0., 0., 1.])
        for n in range(self.N):
            prog.AddQuadraticErrorCost(
                Q=Q,
                x_desired=x_des,
                vars=self.com[:, n]
            ) 
    
    def add_foot_position_cost(self, prog: MathematicalProgram):
        """
        Adds quadratic error cost on foot position decision variables from the reference foot positions
        """
        pass 

    def add_control_cost(self, prog: MathematicalProgram):
        """
        Adds quadratic error cost on control decision variables from the reference control values
        """
        #mg = 10
        Q_force = np.eye(3)*0.001
        Q_force[2, 2] = 0.1
        Q_force_z = np.array([Q_force[2, 2]])
        Q_dforce_dt = 0.001
        Q_com_acc = np.eye(3)*200
        Q_torque = np.eye(3)*2

        def d_force_dt_cost(x: np.ndarray):
            """
            Computes the cost on the rate of change of foot forces
            """
            force_n, force_n_minus_1 = np.split(x, 
                                                [1,])
            force_n /= self.mg
            force_n_minus_1 /= self.mg
            dforce_dt_cost = Q_dforce_dt * np.linalg.norm(force_n - force_n_minus_1, ord=1)
            return dforce_dt_cost

        # def com_acc_cost(x: np.ndarray):
        #     """
        #     Minimize CoM acceleration
        #     """
        #     h, com_vel_k2, com_vel_k1 = np.split(x,
        #                                          [1, 
        #                                           1 + 3,])

        #     com_acc = (com_vel_k2 - com_vel_k1) / h
        #     return 0.5 * (com_acc).T @ Q_com_acc @ (com_acc)

        # for n in range(1, self.N-1):
        #     if self.in_stance[:, n].any() and self.in_stance[:, n-1].any():
        #         com_acc_vars = np.concatenate(
        #             ([self.h[n]],
        #             self.com_dot[:, n],
        #             self.com_dot[:, n-1])
        #         )
        #         prog.AddCost(
        #             com_acc_cost,
        #             vars=com_acc_vars
        #         )

        for n in range(self.N):
            contacts = self.in_stance[:, n]
            num_contacts = np.sum(contacts)
            desired_force_per_contact = self.mg / num_contacts     
            for i in range(len(self.contact_forces)):
                prog.AddQuadraticErrorCost(
                    Q=Q_force_z,
                    x_desired=np.array([desired_force_per_contact]),
                    vars=[self.contact_forces[i][2, n]]
                )
                # prog.AddQuadraticErrorCost(
                #     Q=Q_torque,
                #     x_desired=np.array([0., 0., 0.]),
                #     vars=self.contact_torques[i][:, n]
                # )

            # if n > 0:
            #     for i in range(8):
            #         if self.in_stance[i, n] and self.in_stance[i, n-1]:
            #             d_force_dt_vars = np.array(
            #                 [self.contact_forces[i][2, n], 
            #                     self.contact_forces[i][2, n-1]]
            #             )
            #             # com_acc_vars = np.concatenate(
            #             #     [self.contact_forces[i][:, n],
            #             #     self.contact_forces[i][:, n-1]]
            #             # )
            #             # prog.AddCost(
            #             #     com_acc_cost,
            #             #     vars=com_acc_vars
            #             # )
            #             prog.AddCost(
            #                 d_force_dt_cost,
            #                 vars=d_force_dt_vars
            #             )

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
            prog.AddConstraint(
                UnitQuaternionConstraint(),
                self.body_quat[:, n]
            )

    def add_initial_position_velocity_constraint(self, prog: MathematicalProgram):
        """
        Constrains the initial CoM velocity to be zero
        """

        start_pos_lb = 0.95
        start_pos_ub = 1.05
        prog.AddBoundingBoxConstraint(
            start_pos_lb, 
            start_pos_ub, 
            self.com[2, 0]).evaluator().set_description("initial body pos constraint")

        prog.AddBoundingBoxConstraint(
            start_pos_lb, 
            start_pos_ub, 
            self.com[2, -1]).evaluator().set_description("final body pos constraint")

        # ### test com velocity constraint
        # for n in range(self.N - 1):
        prog.AddBoundingBoxConstraint(
            [0, 0, -0.01],
            [0, 0, 0.01],
            self.com_dot[:, 0]).evaluator().set_description("com velocity constraint t0 ")

        prog.AddBoundingBoxConstraint(
            [0, 0, -0.01],
            [0, 0, 0.01],
            self.com_dot[:, -1]).evaluator().set_description("com velocity constraint tf ")

                           
    def add_com_position_constraint(self, prog: MathematicalProgram):
        """
        Adds direct collocation constraint on CoM positions by integrating CoM velocities
        """
        def com_position_constraint(x: np.ndarray):
            """
            Computes the CoM position constraint from CoM velocity
            """
            h, com_k1, com_dot_k1, foot_forces_k1, foot_forces_k2, com_k2, com_dot_k2 = np.split(x, [
                1,                        # h
                1 + 3,                    # com_k1
                1 + 3 + 3,                # com_dot_k1
                1 + 3 + 3 + 24,           # foot forces k1
                1 + 3 + 3 + 24 + 24,      # foot_forces_k2 
                1 + 3 + 3 + 24 + 24 + 3   # com_k2
            ])
            foot_forces_k1 = foot_forces_k1.reshape((3, 8), order='F')
            foot_forces_k2 = foot_forces_k2.reshape((3, 8), order='F')

            sum_forces_k1 = np.sum(foot_forces_k1, axis=1)
            sum_forces_k2 = np.sum(foot_forces_k2, axis=1)
            com_ddot_k1 = (1/self.mass)*(sum_forces_k1) + self.gravity
            com_ddot_k2 = (1/self.mass)*(sum_forces_k2) + self.gravity
            com_dot_kc = 0.5*(com_dot_k1 + com_dot_k2) + (h/8)*(com_ddot_k1 - com_ddot_k2)
            # direct collocation constraint formula
            rhs = ((-3/(2*h))*(com_k1 - com_k2) - (1/4)*(com_dot_k1 + com_dot_k2))
            return com_dot_kc - rhs 
        
        for n in range(self.N - 1):
            # Define a list of variables to concatenate
            comddot_constraint_variables = [
                [self.h[n]],
                self.com[:, n],
                self.com_dot[:, n],
                *[self.contact_forces[i][:, n] for i in range(8)],
                *[self.contact_forces[i][:, n+1] for i in range(8)],
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

            sum_forces_k1 = np.sum(foot_forces_k1, axis=1)
            sum_forces_k2 = np.sum(foot_forces_k2, axis=1)
            sum_forces_kc = (sum_forces_k1 + sum_forces_k2) / 2

            com_ddot_k1 = (1/self.mass)*sum_forces_k1 + self.gravity
            com_ddot_k2 = (1/self.mass)*sum_forces_k2 + self.gravity
            com_ddot_kc = (1/self.mass)*sum_forces_kc + self.gravity
            # direct collocation constraint formula
            rhs = ((-3/(2*h))*(com_dot_k1 - com_dot_k2) - (0.25)*(com_ddot_k1 + com_ddot_k2))
            return com_ddot_kc - rhs 

        for n in range(self.N - 1):
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

    def add_angular_velocity_constraint(self, prog: MathematicalProgram):
        """
        Integrates the contact torque decision variables to constrain angular velocity decision variables
        """

        # Create autodiff and regular contexts for this constraint (to maximize cache hits)
        ad_angular_velocity_dynamics_context = [
            self.ad_plant.CreateDefaultContext() for i in range(self.N)
        ]

        angular_velocity_dynamics_context = [
            self.plant.CreateDefaultContext() for i in range(self.N)
        ]

        def get_foot_contact_positions(
            srb_yaw_rotation_mat,
            p_W_com,
            p_W_F,
            foot_length,
            foot_width
        ):
            """
            Computes the positions of the four contact points 
            of the left and right foot from the foot conctact location,
            and rotates the positions of the contact points to the body yaw angle d

            p_B_F_W: 3x1 array of the foot contact location measured from the body frame expressed in the world frame
            
            Returns:
                rotated_foot_contact_positions: 3x4 array of the rotated foot contact positions
            """
            p_B_F_W = p_W_F - p_W_com

            p_b_f_w_c1 = np.array([
                p_B_F_W[0] + foot_length/2,
                p_B_F_W[1] + foot_width/2,
                p_B_F_W[2]
            ]).reshape(3, 1)

            p_b_f_w_c2 = np.array([
                p_B_F_W[0] + foot_length/2,
                p_B_F_W[1] - foot_width/2,
                p_B_F_W[2]
            ]).reshape(3, 1)

            p_b_f_w_c3 = np.array([
                p_B_F_W[0] - foot_length/2,
                p_B_F_W[1] + foot_width/2,
                p_B_F_W[2]
            ]).reshape(3, 1)

            p_b_f_w_c4 = np.array([
                p_B_F_W[0] - foot_length/2,
                p_B_F_W[1] - foot_width/2,
                p_B_F_W[2]
            ]).reshape(3, 1)
            foot_contact_positions = np.hstack((p_b_f_w_c1, p_b_f_w_c2, p_b_f_w_c3, p_b_f_w_c4))
            rotated_foot_contact_positions = srb_yaw_rotation_mat@foot_contact_positions
            return rotated_foot_contact_positions

        def compute_omega_dot(
            inertia_mat,
            srb_yaw_rotation_mat,
            p_W_com,
            p_W_LF,
            p_W_RF,
            foot_forces,
            foot_torques,
            foot_length,
            foot_width,
        ):
            """
            computes body angular velocity dynamics omega_dot = f(x,u)
            as omega_dot = I^{-1}*(r_ixF_i + m_i)
            """
            left_foot_forces = foot_forces[:, 0:4]
            right_foot_forces = foot_forces[:, 4:8]

            left_foot_torques = foot_torques[:, 0:4]
            right_foot_torques = foot_torques[:, 4:8]

            # left foot contact positions 
            p_B_LF_contacts = get_foot_contact_positions(
                srb_yaw_rotation_mat,
                p_W_com,
                p_W_LF,
                foot_length,
                foot_width
            )
            # right foot contact positions
            p_B_RF_contacts = get_foot_contact_positions(
                srb_yaw_rotation_mat,
                p_W_com,
                p_W_RF,
                foot_length,
                foot_width
            )

            # compute r x F for left and right foot
            r_x_F_LF = np.cross(p_B_LF_contacts, left_foot_forces, axis=0)
            r_x_F_RF = np.cross(p_B_RF_contacts, right_foot_forces, axis=0)
            sum_forces = np.sum(r_x_F_LF + r_x_F_RF, axis=1).reshape(3, 1)
            sum_torques = np.sum(left_foot_torques + right_foot_torques, axis=1).reshape(3, 1)
            sum_wrench = np.sum(sum_forces + sum_torques, axis=1)
            # return omega dot 
            omega_dot = np.linalg.solve(inertia_mat, sum_wrench)

            return omega_dot


        def angular_velocity_constraint(x: np.ndarray, context_idx: int):
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
                srb_position_state_k1 = np.concatenate((quat_k1, com_k1))
                srb_velocity_state_k1 = np.concatenate((body_angvel_k1, com_dot_k1))
                
                srb_position_state_k2 = np.concatenate((quat_k2, com_k2))
                srb_velocity_state_k2 = np.concatenate((body_angvel_k2, com_dot_k2))

                srb_context_position_state_k1 = self.ad_plant.GetPositions(
                    ad_angular_velocity_dynamics_context[context_idx]
                )
                srb_context_velocity_state_k1 = self.ad_plant.GetVelocities(
                    ad_angular_velocity_dynamics_context[context_idx]
                )

                srb_context_position_state_k2 = self.ad_plant.GetPositions(
                    ad_angular_velocity_dynamics_context[context_idx + 1]
                )
                srb_context_velocity_state_k2 = self.ad_plant.GetVelocities(
                    ad_angular_velocity_dynamics_context[context_idx + 1]
                )

                # Set the context positions and velocities
                if (not autodiff_array_equal(srb_position_state_k1, srb_context_position_state_k1)) or \
                    (not autodiff_array_equal(srb_velocity_state_k1, srb_context_velocity_state_k1)):
                    self.ad_plant.SetPositionsAndVelocities(
                        ad_angular_velocity_dynamics_context[context_idx],
                        np.concatenate((srb_position_state_k1, srb_velocity_state_k1))
                    )
                if (not autodiff_array_equal(srb_position_state_k2, srb_context_position_state_k2)) or \
                    (not autodiff_array_equal(srb_velocity_state_k2, srb_context_velocity_state_k2)):
                    self.ad_plant.SetPositionsAndVelocities(
                        ad_angular_velocity_dynamics_context[context_idx + 1],
                        np.concatenate((srb_position_state_k2, srb_velocity_state_k2))
                    )
                
                I_B_W_k1 = self.ad_plant.CalcSpatialInertia(
                    ad_angular_velocity_dynamics_context[context_idx],
                    self.ad_plant.world_frame(),
                    [self.srb_body_idx]
                ).CalcRotationalInertia().CopyToFullMatrix3()

                I_B_W_k2 = self.ad_plant.CalcSpatialInertia(
                    ad_angular_velocity_dynamics_context[context_idx + 1],
                    self.ad_plant.world_frame(),
                    [self.srb_body_idx]
                ).CalcRotationalInertia().CopyToFullMatrix3()
                srb_orientation_euler_k1 = self.ad_plant.EvalBodyPoseInWorld(
                    ad_angular_velocity_dynamics_context[context_idx],
                    self.ad_srb_body
                ).rotation().ToRollPitchYaw().vector()

                srb_yaw_rotation_k1 = RotationMatrix_[AutoDiffXd].MakeZRotation(srb_orientation_euler_k1[2]).matrix()
                q_dot_k1 = np.zeros_like(srb_context_position_state_k1)
                q_dot_k2 = np.zeros_like(srb_context_position_state_k2)
                q_dot_k1 = self.ad_plant.MapVelocityToQDot(
                    ad_angular_velocity_dynamics_context[context_idx],
                    self.ad_plant.GetVelocities(ad_angular_velocity_dynamics_context[context_idx]),
                )
                q_dot_k2 = self.ad_plant.MapVelocityToQDot(
                    ad_angular_velocity_dynamics_context[context_idx + 1],
                    self.ad_plant.GetVelocities(ad_angular_velocity_dynamics_context[context_idx + 1]),
                )

                quat_dot_k1 = q_dot_k1[0:4]
                quat_dot_k2 = q_dot_k2[0:4]


                k1_variables = [
                    I_B_W_k1.reshape(3,3),
                    srb_yaw_rotation_k1.reshape(3,3),
                    com_k1.reshape(3,),
                    p_W_LF_k1.reshape(3,),
                    p_W_RF_k1.reshape(3,),
                    foot_forces_k1,
                    foot_torques_k1,
                ]
                k1_values = [ExtractValue(var) for var in k1_variables]
                k1_grads = [ExtractGradient(var) for var in k1_variables]
                
                k2_variables = [
                    I_B_W_k2.reshape(3,3),
                    srb_yaw_rotation_k1.reshape(3,3),
                    com_k2.reshape(3,),
                    p_W_LF_k2.reshape(3,),
                    p_W_RF_k2.reshape(3,),
                    foot_forces_k2,
                    foot_torques_k2,
                ]
                k2_values = [ExtractValue(var) for var in k2_variables]
                k2_grads = [ExtractGradient(var) for var in k2_variables]
                
                omega_dot_k1, omega_dot_k1_jac = compute_omega_dot_jax(
                    *k1_values,
                    self.options.foot_length,
                    self.options.foot_width
                )
                #omega_dot_k1_jac = [omega_dot_k1[i]*scale for i in range(len(omega_dot_k1_jac))]
                #omega_dot_k1_jac *= scale

                # Use map to apply the function to each pair of matrices
                k1_chain_rule_matrices = map(multiply_and_sum, zip(omega_dot_k1_jac, k1_grads))
                # Convert the result_matrices to a NumPy array and sum along the 3rd axis
                del_omega_dot_del_x = np.sum(np.array(list(k1_chain_rule_matrices)), axis=0)
                omega_dot_k1 = InitializeAutoDiff(value=omega_dot_k1, gradient=del_omega_dot_del_x).reshape(3,)

                omega_dot_k2, omega_dot_k2_jac = compute_omega_dot_jax(
                    *k2_values,
                    self.options.foot_length,
                    self.options.foot_width
                )
                #omega_dot_k2_jac = [omega_dot_k2[i]*scale for i in range(len(omega_dot_k2_jac))]
                #omega_dot_k2_jac *= 1/scale

                k2_chain_rule_matrices = map(multiply_and_sum, zip(omega_dot_k2_jac, k2_grads))
                del_omega_dot_del_x = np.sum(np.array(list(k2_chain_rule_matrices)), axis=0)
                omega_dot_k2 = InitializeAutoDiff(value=omega_dot_k2, gradient=del_omega_dot_del_x).reshape(3,)

                foot_forces_kc = np.array([foot_forces_k1, foot_forces_k2]).mean(axis=0)
                foot_torques_kc = np.array([foot_torques_k1, foot_torques_k2]).mean(axis=0)

                # foot positions in stance should not change, so p_W_F_kc = p_W_F_k1 = p_W_F_k2
                p_W_LF_kc = np.array([p_W_LF_k1, p_W_LF_k2]).mean(axis=0)
                p_W_RF_kc = np.array([p_W_RF_k1, p_W_RF_k2]).mean(axis=0)
                # initial body orientation quat_k1 is used for all dynamics evaluations 
                # because the feet in contact do not move 
                # compute com_kc 
                com_kc = 0.5*(com_k1 + com_k2) + (h/8)*(com_dot_k1 - com_dot_k2)
                # quat_kc 
                # interpolate the quaternion at the midpoint using SLERP
                zero_ad = AutoDiffXd(0)
                # brute force normalize quaternions 
                quat_k1 = quat_k1 / np.linalg.norm(quat_k1)
                quat_k2 = quat_k2 / np.linalg.norm(quat_k2) 
                drake_quat_k1 = Quaternion_[AutoDiffXd](quat_k1)
                drake_quat_k2 = Quaternion_[AutoDiffXd](quat_k2)
                #zero_ad = InitializeAutoDiff(value=np.array([0.]), gradient=np.array([0.]))
                slerp = PiecewiseQuaternionSlerp_[AutoDiffXd]([zero_ad, h.item()],
                                                              [drake_quat_k1, drake_quat_k2]) 
                
                quat_kc = slerp.orientation((h/2).item()).wxyz() + (h/8)*(quat_dot_k1 - quat_dot_k2)
                # quat_kc = 0.5*(quat_k1 + quat_k2) + (h/8)* \
                #     (quat_dot_k1 - quat_dot_k2)
                # # normalize quat_kc
                quat_kc = quat_kc / np.linalg.norm(quat_kc)
                
                srb_kc_context = self.ad_plant.CreateDefaultContext()
                self.ad_plant.SetPositions(
                    srb_kc_context,
                    np.concatenate((quat_kc, com_kc))
                )
                I_B_W_kc = self.ad_plant.CalcSpatialInertia(
                    srb_kc_context,
                    self.ad_plant.world_frame(),
                    [self.srb_body_idx]
                ).CalcRotationalInertia().CopyToFullMatrix3()
                kc_variables = [
                    I_B_W_kc.reshape(3,3),
                    srb_yaw_rotation_k1.reshape(3,3),
                    com_kc.reshape(3,),
                    p_W_LF_kc.reshape(3,),
                    p_W_RF_kc.reshape(3,),
                    foot_forces_kc,
                    foot_torques_kc,
                ]
                kc_values = [ExtractValue(var) for var in kc_variables]
                kc_grads = [ExtractGradient(var) for var in kc_variables]

                omega_dot_kc, omega_dot_kc_jac = compute_omega_dot_jax(
                    *kc_values,
                    self.options.foot_length,
                    self.options.foot_width
                )
                # omega_dot_kc_jac = [omega_dot_kc[i]*scale for i in range(len(omega_dot_kc_jac))]
                # #omega_dot_kc_jac *= scale
                # #omega_dot_kc_jac[-2] *= 1/scale

                kc_chain_rule_matrices = map(multiply_and_sum, zip(omega_dot_kc_jac, kc_grads))
                del_omega_dot_del_x = np.sum(np.array(list(kc_chain_rule_matrices)), axis=0)
                omega_dot_kc = InitializeAutoDiff(value=omega_dot_kc, gradient=del_omega_dot_del_x).reshape(3,)
                # print(omega_dot_kc.shape)
                # direct collocation constraint formula
                rhs = ((-3/(2*h))*(body_angvel_k1 - body_angvel_k2) - (1/4)*(omega_dot_k1 + omega_dot_k2))

                return omega_dot_kc - rhs

            else:   
                srb_position_state_k1 = np.concatenate((quat_k1, com_k1))
                srb_velocity_state_k1 = np.concatenate((body_angvel_k1, com_dot_k1))
                
                srb_position_state_k2 = np.concatenate((quat_k2, com_k2))
                srb_velocity_state_k2 = np.concatenate((body_angvel_k2, com_dot_k2))

                srb_context_position_state_k1 = self.ad_plant.GetPositions(
                    ad_angular_velocity_dynamics_context[context_idx]
                )
                srb_context_velocity_state_k1 = self.ad_plant.GetVelocities(
                    ad_angular_velocity_dynamics_context[context_idx]
                )

                srb_context_position_state_k2 = self.ad_plant.GetPositions(
                    ad_angular_velocity_dynamics_context[context_idx + 1]
                )
                srb_context_velocity_state_k2 = self.ad_plant.GetVelocities(
                    ad_angular_velocity_dynamics_context[context_idx + 1]
                )
                # Set the context positions and velocities
                if (not np.array_equal(srb_position_state_k1, srb_context_position_state_k1)) or \
                    (not np.array_equal(srb_velocity_state_k1, srb_context_velocity_state_k1)):
                    self.plant.SetPositionsAndVelocities(
                        angular_velocity_dynamics_context[context_idx],
                        np.concatenate((srb_position_state_k1, srb_velocity_state_k1))
                    )
                if (not np.array_equal(srb_position_state_k2, srb_context_position_state_k2)) or \
                    (not np.array_equal(srb_velocity_state_k2, srb_context_velocity_state_k2)):
                    self.plant.SetPositionsAndVelocities(
                        angular_velocity_dynamics_context[context_idx + 1],
                        np.concatenate((srb_position_state_k2, srb_velocity_state_k2))
                    )
                
                I_B_W_k1 = self.plant.CalcSpatialInertia(
                    angular_velocity_dynamics_context[context_idx],
                    self.plant.world_frame(),
                    [self.srb_body_idx]
                ).CalcRotationalInertia().CopyToFullMatrix3()

                I_B_W_k2 = self.plant.CalcSpatialInertia(
                    angular_velocity_dynamics_context[context_idx + 1],
                    self.plant.world_frame(),
                    [self.srb_body_idx]
                ).CalcRotationalInertia().CopyToFullMatrix3()
                srb_orientation_euler_k1 = self.plant.EvalBodyPoseInWorld(
                    angular_velocity_dynamics_context[context_idx],
                    self.srb_body
                ).rotation().ToRollPitchYaw().vector()

                srb_yaw_rotation_k1 = RotationMatrix.MakeZRotation(srb_orientation_euler_k1[2]).matrix()
                q_dot_k1 = np.zeros_like(srb_context_position_state_k1)
                q_dot_k2 = np.zeros_like(srb_context_position_state_k2)
                q_dot_k1 = self.plant.MapVelocityToQDot(
                    angular_velocity_dynamics_context[context_idx],
                    self.plant.GetVelocities(angular_velocity_dynamics_context[context_idx]),
                )
                q_dot_k2 = self.plant.MapVelocityToQDot(
                    angular_velocity_dynamics_context[context_idx + 1],
                    self.plant.GetVelocities(angular_velocity_dynamics_context[context_idx + 1]),
                )

                quat_dot_k1 = q_dot_k1[0:4]
                quat_dot_k2 = q_dot_k2[0:4]
                k1_variables = [
                    I_B_W_k1.reshape(3,3),
                    srb_yaw_rotation_k1.reshape(3,3),
                    com_k1.reshape(3,),
                    p_W_LF_k1.reshape(3,),
                    p_W_RF_k1.reshape(3,),
                    foot_forces_k1,
                    foot_torques_k1,
                ]

                k2_variables = [
                    I_B_W_k2.reshape(3,3),
                    srb_yaw_rotation_k1.reshape(3,3),
                    com_k2.reshape(3,),
                    p_W_LF_k2.reshape(3,),
                    p_W_RF_k2.reshape(3,),
                    foot_forces_k2,
                    foot_torques_k2,
                ]

                omega_dot_k1 = compute_omega_dot(
                    *k1_variables,
                    self.options.foot_length,
                    self.options.foot_width
                )

                omega_dot_k2 = compute_omega_dot(
                    *k2_variables,
                    self.options.foot_length,
                    self.options.foot_width
                )

                foot_forces_kc = np.array([foot_forces_k1, foot_forces_k2]).mean(axis=0)
                foot_torques_kc = np.array([foot_torques_k1, foot_torques_k2]).mean(axis=0)

                # foot positions in stance should not change, so p_W_F_kc = p_W_F_k1 = p_W_F_k2
                p_W_LF_kc = np.array([p_W_LF_k1, p_W_LF_k2]).mean(axis=0)
                p_W_RF_kc = np.array([p_W_RF_k1, p_W_RF_k2]).mean(axis=0)
                # initial body orientation quat_k1 is used for all dynamics evaluations 
                # because the feet in contact do not move 
                # compute com_kc 
                com_kc = 0.5*(com_k1 + com_k2) + (h/8)*(com_dot_k1 - com_dot_k2)
                # brute force normalize quaternions 
                quat_k1 = quat_k1 / np.linalg.norm(quat_k1)
                quat_k2 = quat_k2 / np.linalg.norm(quat_k2) 
                # interpolate the quaternion at the midpoint using SLERP
                drake_quat_k1 = Quaternion(quat_k1)
                drake_quat_k2 = Quaternion(quat_k2)
                #zero_ad = InitializeAutoDiff(value=np.array([0.]), gradient=np.array([0.]))
                slerp = PiecewiseQuaternionSlerp([0., h[0]],
                                                 [drake_quat_k1, drake_quat_k2]) 
                
                quat_kc = slerp.orientation((h[0]/2)).wxyz() + (h[0]/8)*(quat_dot_k1 - quat_dot_k2)
                quat_kc = quat_kc / np.linalg.norm(quat_kc)
                
                srb_kc_context = self.plant.CreateDefaultContext()
                self.plant.SetPositions(
                    srb_kc_context,
                    np.concatenate((quat_kc, com_kc))
                )
                I_B_W_kc = self.plant.CalcSpatialInertia(
                    srb_kc_context,
                    self.plant.world_frame(),
                    [self.srb_body_idx]
                ).CalcRotationalInertia().CopyToFullMatrix3()
                kc_variables = [
                    I_B_W_kc.reshape(3,3),
                    srb_yaw_rotation_k1.reshape(3,3),
                    com_kc.reshape(3,),
                    p_W_LF_kc.reshape(3,),
                    p_W_RF_kc.reshape(3,),
                    foot_forces_kc,
                    foot_torques_kc,
                ]

                omega_dot_kc = compute_omega_dot(
                    *kc_variables,
                    self.options.foot_length,
                    self.options.foot_width
                )

                # direct collocation constraint formula
                rhs = ((-3/(2*h))*(body_angvel_k1 - body_angvel_k2) - (1/4)*(omega_dot_k1 + omega_dot_k2))
                return omega_dot_kc - rhs

        for n in range(self.N - 1):
            # Define a list of variables to concatenate
            angvel_constraint_variables = [
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
            
            flattened_variables = [item for sublist in angvel_constraint_variables for item in sublist]

            prog.AddConstraint(
                partial(angular_velocity_constraint, context_idx=n),
                lb=[0,0,0],
                ub=[0,0,0],
                vars=np.array(flattened_variables),
                description=f"angular_velocity_constraint_{n}"
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
        max_z_grf = float(self.options.max_z_grf)
        foot_half_l = float(self.options.foot_length/2)
        foot_half_w = float(self.options.foot_width/2)

        for n in range(self.N):
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
                    self.in_stance[i, n]*max_z_grf,
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
            # #############################################


            # # M_x max torque constraints
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

                self.add_linear_absolute_value_constraint(
                    prog,
                    p_B_LF_W_i,
                    self.options.leg_extension_bounds[i]
                )
                self.add_linear_absolute_value_constraint(
                    prog,
                    p_B_RF_W_i,
                    self.options.leg_extension_bounds[i]
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
                if self.in_stance[i, n]:
                    if i in range(4):
                        prog.AddLinearConstraint(
                            self.p_W_LF[2,n] == 0
                        ).evaluator().set_description(f"foot_{i}_stance_z_pos_constraint_{n}")

                        if n > 0 and self.in_stance[i, n-1]:
                            # feet should not move during stance 
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
                    elif i in range(4, 8):
                        prog.AddLinearConstraint(
                            self.p_W_RF[2,n] == 0
                        ).evaluator().set_description(f"foot_{i}_stance_z_pos_constraint_{n}")
                        if n > 0 and self.in_stance[i, n-1]:
                            # feet should not move during stance 
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
                else:
                    if i in range(4):
                        prog.AddLinearConstraint(
                            self.p_W_LF[2,n] >= 0.001
                        ).evaluator().set_description(f"foot_{i}_flight_z_pos_constraint_{n}")
                    elif i in range(4, 8):
                        prog.AddLinearConstraint(
                            self.p_W_RF[2,n] >= 0.001
                        ).evaluator().set_description(f"foot_{i}_flight_z_pos_constraint_{n}")

    def add_minimum_com_height_constraint(self, prog: MathematicalProgram):
        """
        Constrains the minimum com height to be above a certain threshold
        """
        for n in range(self.N):
            prog.AddLinearConstraint(
                self.com[2, n] >= self.options.min_com_height
            ).evaluator().set_description(f"min_com_height_constraint_{n}")

    def formulate_trajopt_problem(self,):
        """
        Formulates the trajectory optimization problem by defining all costs and constraints
        """
        prog = self.create_trajopt_program()
        self.create_contact_sequence()
        self.set_trajopt_initial_guess(prog)

        ### Add Costs ###
        self.add_com_position_cost(prog)
        self.add_control_cost(prog)

        ### Add Constraints ###
        self.add_time_scaling_constraint(prog)
        self.add_unit_quaternion_constraint(prog)
        self.add_quaternion_integration_constraint(prog)
        self.add_com_velocity_constraint(prog)
        self.add_com_position_constraint(prog)
        
        self.add_initial_position_velocity_constraint(prog)
        self.add_step_length_kinematic_constraint(prog)
        self.add_angular_velocity_constraint(prog)
        self.add_contact_wrench_cone_constraint(prog)
        self.add_foot_velocity_kinematic_constraint(prog)
        self.add_minimum_com_height_constraint(prog)

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
        self.configure_snopt_solver(prog)
        # scale contact force/torque decision variables
        for i in range(len(self.contact_forces)):
            for j in range(3):
                for k in range(len(self.contact_forces[i][j])):
                    prog.SetVariableScaling(self.contact_forces[i][j, k], self.mg)
                    # prog.SetVariableScaling(self.contact_torques[i][j, k], mg)

        res = Solve(prog)
        quat = res.GetSolution(self.body_quat)
        print(res.is_success())
        print(res.get_solver_details().info)
        print(f"solution cost: {res.get_optimal_cost()}")
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
        left_foot_torque_z_sum = []
        right_foot_torque_z_sum = []
        # for i in range(4):
        #     contact_frc_soln = res.GetSolution(self.contact_forces[i])
        for i in range(self.N-1):
            contact_frc_sum = 0
            contact_trq_sum = 0
            for j in range(4):
                contact_frc_soln = res.GetSolution(self.contact_forces[j])
                contact_torque_soln = res.GetSolution(self.contact_torques[j])
                contact_frc_sum += contact_frc_soln[2, i]
                contact_trq_sum += contact_torque_soln[1, i]
            left_foot_torque_z_sum.append(contact_trq_sum)
            left_foot_z_frc_sum.append(contact_frc_sum)
            
        for i in range(self.N-1):
            contact_frc_sum = 0
            contact_trq_sum = 0
            for j in range(4, 8):
                contact_frc_soln = res.GetSolution(self.contact_forces[j])
                contact_torque_soln = res.GetSolution(self.contact_torques[j])
                contact_frc_sum += contact_frc_soln[2, i]
                contact_trq_sum += contact_torque_soln[1, i]
            right_foot_z_frc_sum.append(contact_frc_sum)
            right_foot_torque_z_sum.append(contact_trq_sum)
        
        time = np.cumsum(np.hstack((0, res.GetSolution(self.h))))
        plt.plot(time, res.GetSolution(self.com)[2, :])
        plt.show()
        plt.plot(time[:-1], left_foot_z_frc_sum)
        plt.plot(time[:-1], right_foot_z_frc_sum)
        plt.show()
        plt.plot(time[:-1], left_foot_torque_z_sum)
        plt.plot(time[:-1], right_foot_torque_z_sum)
        plt.show()
        # print(res.GetSolution(self.contact_forces[0]))
        # print(res.GetSolution(self.com_dot))

        return res

    def create_contact_sequence(self):

        N = self.options.N
        # use placeholder walking gait contact sequnce for now 
        in_stance = np.zeros((8, N))
        in_stance[0:4, 0:int(N/2)] = 1
        in_stance[4:8, int(N/2)-1:N] = 1

        
        # in_stance = np.ones((8, N))
        # in_stance[0:4, int(N/3):int(3*N/4)] = 0
        # in_stance[4:8, int(N/3):int(3*N/4)] = 0

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




