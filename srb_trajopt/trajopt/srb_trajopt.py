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
    Simulator_
)

from pydrake.math import (
    RigidTransform
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
    Solve)

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
)
from .jax_utils import (
    np_to_jax,
    batch_np_to_jax,
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
        self.meshcat = srb_builder.meshcat
        self.body_v0 = np.array([0., 0., 2.]) # initial body velocity
        self.ad_simulator = Simulator_[AutoDiffXd](self.ad_srb_diagram, 
                                      self.ad_srb_diagram.CreateDefaultContext())
        self.ad_simulator.Initialize()
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
        self.p_B_LF_W = prog.NewContinuousVariables(3, self.N, "left_foot_pos")
        self.p_B_RF_W = prog.NewContinuousVariables(3, self.N, "right_foot_pos")

        #####################################
        # define control decision variables #
        #####################################

        foot_wrench_names = ["left_front_left", 
                             "left_front_right", 
                             "left_back_left", 
                             "left_back_right",
                             "right_front_left",
                             "right_front_right",
                             "right_back_left",
                             "right_back_right"
                             ]
        contact_forces = [
            prog.NewContinuousVariables(3, self.N - 1, f"foot_{foot}_force") for foot in foot_wrench_names
        ]
        contact_torques = [
            prog.NewContinuousVariables(3, self.N - 1, f"foot_{foot}_torque") for foot in foot_wrench_names
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
                constraint_val, flattened_constraint_jac = com_dircol_constraint_jit(
                    h_val.item(),
                    com_k1_val,
                    com_dot_k1_val,
                    foot_forces_k1_val,
                    com_k2_val,
                    com_dot_k2_val,
                    self.mass,
                    self.gravity)

                constraint_val_ad = InitializeAutoDiff(value=constraint_val,
                                                       gradient=flattened_constraint_jac)
                return constraint_val_ad
            else:
                sum_forces_k1 = np.sum(foot_forces_k1, axis=1)
                com_ddot_k1 = (sum_forces_k1 / self.mass) + self.gravity
                com_dot_kc = com_dot_k1 + (h/2)*com_ddot_k1
                # direct collocation constraint formula
                rhs = (-3/(2*h))*(com_k1 - com_k2) - (1/4)*(com_dot_k1 + com_dot_k2)
                return com_dot_kc - rhs 
        
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
            p_ddot = 1/m * sum(F_i) + g
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
                constraint_val, flattened_constraint_jac = com_dot_dircol_constraint_jit(
                    h_val.item(),
                    com_dot_k1_val,
                    foot_forces_k1_val,
                    com_dot_k2_val,
                    foot_forces_k2_val,
                    self.mass,
                    self.gravity)

                constraint_val_ad = InitializeAutoDiff(value=constraint_val,
                                                       gradient=flattened_constraint_jac)
                return constraint_val_ad
            else:
                sum_forces_k1 = np.sum(foot_forces_k1, axis=1)
                sum_forces_k2 = np.sum(foot_forces_k2, axis=1)
                sum_forces_k_c = (sum_forces_k1 + sum_forces_k2) / 2
                com_ddot_k1 = (sum_forces_k1 / self.mass) + self.gravity
                com_ddot_k2 = (sum_forces_k2 / self.mass) + self.gravity
                com_ddot_kc = (sum_forces_k_c / self.mass) + self.gravity
                # direct collocation constraint formula
                rhs = (-3/(2*h))*(com_dot_k1 - com_dot_k2) - (1/4)*(com_ddot_k1 + com_ddot_k2)
                return com_ddot_kc - rhs 

        
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

        def get_srb_inertia_mat_from_context(context: Context):
                spatial_inertia_k1 = self.ad_plant.CalcSpatialInertia(
                    context,
                    self.ad_plant.world_frame(), 
                    [self.srb_body_idx]
                )
                rot_inertia = spatial_inertia_k1.CalcRotationalInertia()
                inertia_mat = rot_inertia.CopyToFullMatrix3()

                return inertia_mat

        def angular_velocity_constraint(x: np.ndarray, context_index: int):
            # Define variable names and their corresponding indices
            var_names = {
                'h': 1,
                'com_k1': 4,
                'quat_k1': 8,
                'com_dot_k1': 11,
                'body_angvel_k1': 14,
                'p_lf_k1': 17,
                'p_rf_k1': 20,
                'foot_forces_k1': 44,
                'foot_torques_k1': 68,
                'com_k2': 71,
                'body_angvel_k2': 74,
                'p_lf_k2': 77,
                'p_rf_k2': 80,
                'foot_forces_k2': 104,
                'foot_torques_k2': 128,
            }
            # Split the input array using the indices
            split_vars = np.split(x, [
                1,                   # h
                4,                   # com_k1
                8,                   # quat_k1
                11,                  # com_dot_k1
                14,                  # body_angvel_k1
                17,                  # p_lf_k1
                20,                  # p_rf_k1
                44,                  # foot_forces_k1
                68,                  # foot_torques_k1
                71,                  # com_k2
                74,                  # body_angvel_k2
                77,                  # p_lf_k2
                80,                  # p_rf_k2
                104,                 # foot_forces_k2
            ])
            # Unpack the split variables into meaningful names
            h, com_k1, quat_k1, com_dot_k1, body_angvel_k1, p_lf_k1, p_rf_k1, \
            foot_forces_k1, foot_torques_k1, com_k2, body_angvel_k2, \
            p_lf_k2, p_rf_k2, foot_forces_k2, foot_torques_k2 = split_vars

            if isinstance(x[0], AutoDiffXd):
                srb_k1_pos = np.concatenate([quat_k1, com_k1])
                srb_k1_vel = np.concatenate([body_angvel_k1, com_dot_k1])
                srb_pos_equal = autoDiffArrayEqual(
                    srb_k1_pos,
                    self.ad_plant.GetPositions(
                        ad_angular_velocity_dynamics_context[context_index]
                    ),
                )
                srb_vel_equal = autoDiffArrayEqual(
                    srb_k1_vel,
                    self.ad_plant.GetVelocities(
                        ad_angular_velocity_dynamics_context[context_index]
                    ),
                )
                if not (srb_pos_equal or srb_vel_equal):
                    self.ad_plant.SetPositions(
                        ad_angular_velocity_dynamics_context[context_index], srb_k1_pos
                    )
                    self.ad_plant.SetVelocities(
                        ad_angular_velocity_dynamics_context[context_index], srb_k1_vel
                    )
                
                inertia_mat_k1 = get_srb_inertia_mat_from_context(
                    ad_angular_velocity_dynamics_context[context_index]
                )

                inertia_mat_k1_val, del_inertia_mat_k1_del_x = extract_ad_value_and_gradient(
                    inertia_mat_k1
                )
                print(inertia_mat_k1)
                
                # set simulator context state 
                sim_context = self.ad_simulator.get_mutable_context()
                sim_context.SetTime(0.)
                sim_context.SetContinuousState(
                    ad_angular_velocity_dynamics_context[context_index].get_continuous_state().CopyToVector()
                )

                collocation_time = h.item()/2
                self.ad_simulator.AdvanceTo(collocation_time)
                sim_state = sim_context.get_continuous_state().CopyToVector()
                ad_angular_velocity_dynamics_context[context_index].SetContinuousState(sim_state)

                inertia_mat_kc = get_srb_inertia_mat_from_context(
                    ad_angular_velocity_dynamics_context[context_index]
                )

                inertia_mat_kc_val, del_inertia_mat_kc_del_x = extract_ad_value_and_gradient(
                    inertia_mat_kc
                )
                t2 = h.item()
                self.ad_simulator.AdvanceTo(t2)

                # set velocity dynamics context state to simulator context state
                sim_state = sim_context.get_continuous_state().CopyToVector()
                ad_angular_velocity_dynamics_context[context_index].SetContinuousState(sim_state)
                inertia_mat_k2 = get_srb_inertia_mat_from_context(
                    ad_angular_velocity_dynamics_context[context_index]
                )


            else:   
            
                pass 
        
        for n in range(self.N - 2):
            # Define a list of variables to concatenate
            comddot_constraint_variables = [
                [self.h[n]],
                self.com[:, n],
                self.body_quat[:, n],
                self.com_dot[:, n],
                self.body_angvel[:, n],
                self.p_B_LF_W[:, n],
                self.p_B_RF_W[:, n],
                *[self.contact_forces[i][:, n] for i in range(8)],
                *[self.contact_torques[i][:, n] for i in range(8)],
                self.com[:, n + 1],
                self.body_angvel[:, n + 1],
                self.p_B_LF_W[:, n + 1],
                self.p_B_RF_W[:, n + 1],
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

    def add_contact_wrench_cone_constraint(self, prog: MathematicalProgram):
        """
        Constrains the contact wrench decision variables to lie within the wrench cone set 
        
        """

        pass 

    def add_step_length_kinematic_constraint(self, prog: MathematicalProgram):
        """
        Constrains the step location and leg extension to be within kinematically feasible bounds
        """
        pass 
    
    def formulate_trajopt_problem(self,):
        """
        Formulates the trajectory optimization problem by defining all costs and constraints
        """
        prog = self.create_trajopt_program()
        self.set_trajopt_initial_guess(prog)
        self.add_time_scaling_constraint(prog)
        self.add_unit_quaternion_constraint(prog)
        self.add_quaternion_integration_constraint(prog)
        self.add_com_velocity_constraint(prog)
        self.add_com_position_constraint(prog)
        self.add_initial_velocity_constraint(prog)
        self.add_angular_velocity_constraint(prog)

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
        res = Solve(prog)
        print(res.get_solution_result())
        # print(res.GetSolution(self.h))
        # print(np.sum(res.GetSolution(self.h)))
        # print(res.GetSolution(self.body_quat))
        #print(res.GetSolution(self.com))
        # plot result 
        time = np.cumsum(np.hstack((0, res.GetSolution(self.h))))
        # plt.plot(time, res.GetSolution(self.com)[2, :])
        # plt.show()
        # print(res.GetSolution(self.contact_forces[0]))
        # print(res.GetSolution(self.com_dot))

        return res

    def create_contact_sequence(self):

        N = self.options.N
        # use placeholder walking gait contact sequnce for now 
        in_stance = np.zeros((4, N))
        in_stance[0:4, 0:int(N/2)] = 1
        in_stance[4:8, int(N/2):N] = 1
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




