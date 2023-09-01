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

import numpy as np
import time

class SRBTrajopt:
    def __init__(self, 
                 options: SRBTrajoptOptions,
                 headless: bool = False) -> None:
        self.options = options
        srb_builder = SRBBuilder(self.options, headless)
        self.srb_diagram = srb_builder.create_srb_diagram()
        self.meshcat = srb_builder.meshcat

    @property
    def plant(self) -> MultibodyPlant:
        """
        Returns:
            plant: MultibodyPlant object from SRB diagram
        """
        return self.srb_diagram.GetSubsystemByName("plant")

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
        N = self.options.N
        # define state decision variables
        prog.NewContinuousVariables(N - 1, "h")
        prog.NewContinuousVariables(3, N, "com_pos")
        prog.NewContinuousVariables(3, N, "com_vel")
        prog.NewContinuousVariables(4, N, "body_quat")
        prog.NewContinuousVariables(3, N, "body_angular_vel")
        prog.NewContinuousVariables(3, N, "left_foot_pos")
        prog.NewContinuousVariables(3, N, "right_foot_pos")
        # define control decision variables
        foot_wrench_names = ["front_left", "front_right", "back_left", "back_right"]

        contact_forces = [
            prog.NewContinuousVariables(3, N - 1, f"foot_{foot}_force") for foot in foot_wrench_names
        ]
        contact_torques = [
            prog.NewContinuousVariables(3, N - 1, f"foot_{foot}_torque") for foot in foot_wrench_names
        ]
        self.contact_forces = contact_forces
        self.contact_torques = contact_torques
        return prog

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




