from options import SRBTrajoptOptions
from .srb_builder import SRBBuilder
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
    com_ddot_constraint_jit,
)
from .jax_utils import (
    np_to_jax,
    batch_np_to_jax,
)
class SRBTrajopt:
    def __init__(self, 
                 options: SRBTrajoptOptions,
                 headless: bool = False) -> None:
        self.options = options
        srb_builder = SRBBuilder(self.options, headless)
        self.srb_diagram, self.ad_srb_diagram = srb_builder.create_srb_diagram()
        self.meshcat = srb_builder.meshcat

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
        self.com_ddot = prog.NewContinuousVariables(3, self.N-1, "com_ddot")

        self.body_quat = prog.NewContinuousVariables(4, self.N, "body_quat")
        self.body_angvel = prog.NewContinuousVariables(3, self.N, "body_angular_vel")
        self.p_w_lf = prog.NewContinuousVariables(3, self.N, "left_foot_pos")
        self.p_w_rf = prog.NewContinuousVariables(3, self.N, "right_foot_pos")

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
        default_angvel = np.array([0.0, 0.0, 0.1])
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
            if n < self.N - 1:
                # set CoM acceleration guess
                prog.SetInitialGuess(
                    self.com_ddot[:, n],
                    default_com_ddot
                )
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

    def add_quaternion_integration_constraint(self, prog: MathematicalProgram):
        """
        Add euler quaternion integration constraint to ensure that q1 rotates to q2 in dt time for a 
        given angular velocity w assuming that the angular velocity is constant over the time interval
        """
        for n in range(self.N - 1):
            # Define symbolic variables
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
        Integrates the CoM velocity decision variables to constrain CoM position decision variables
        """
        pass 
    
    def add_com_acceleration_constraint(self, prog: MathematicalProgram):
        """
        Constrains the CoM acceleration decision variables to be consistent with the contact forces
        """
        def com_acceleration_constraint(x: np.ndarray,):
            """
            Computes the CoM velocity constraint for a given CoM velocity and contact forces
            p_ddot = 1/m * sum(F_i) + g
            """
            com_ddot, left_foot_forces, right_foot_forces = np.split(x, [3, # com_ddot 
                                                                         3 + 12, # left foot
                                                                         ])
            
            left_foot_forces = left_foot_forces.reshape((3, 4), order='F')
            right_foot_forces = right_foot_forces.reshape((3, 4), order='F')
            foot_forces = np.concatenate((left_foot_forces, right_foot_forces), axis=1)
            # Check if the first element of x is an AutoDiffXd object
            if isinstance(x[0], AutoDiffXd):                
                # Extract values from autodiff decision variables
                com_ddot_val = ExtractValue(com_ddot).reshape(3,)
                foot_forces_val = ExtractValue(foot_forces)
                # Compute the constraint value and gradients using JAX
                constraint_val, flattened_constraint_jac = com_ddot_constraint_jit(
                    com_ddot_val,
                    foot_forces_val,
                    self.mass,
                    self.gravity)
                constraint_val_ad = InitializeAutoDiff(value=constraint_val,gradient=flattened_constraint_jac)
                return constraint_val_ad
            else:
                foot_forces = np.concatenate((left_foot_forces, right_foot_forces), axis=1)
                sum_forces = np.sum(foot_forces, axis=1)
                mg = self.mass * self.gravity[2]
                acc = np.array([0., 0., 1.]) + (sum_forces/mg)
                return acc - com_ddot/self.gravity[2]
        
        for n in range(self.N - 1):
            # Define a list of variables to concatenate
            comddot_constraint_variables = [
                self.com_ddot[:, n],
                *[self.contact_forces[i][:, n] for i in range(8)]]
            flattened_variables = [item for sublist in comddot_constraint_variables for item in sublist]
            
            prog.AddConstraint(
                com_acceleration_constraint,
                lb=[0,0,0],
                ub=[0,0,0],
                vars=np.array(flattened_variables),
                description=f"com_acceleration_constraint_{n}"
            )

    def add_angular_velocity_constraint(self, prog: MathematicalProgram):
        """
        Integrates the contact torque decision variables to constrain angular velocity decision variables
        """
        pass

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
        self.add_com_acceleration_constraint(prog)

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
        print(res.GetSolution(self.com_ddot))

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




