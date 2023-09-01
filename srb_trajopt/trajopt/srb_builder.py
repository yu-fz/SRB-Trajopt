from options import SRBTrajoptOptions
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

class SRBBuilder:
    def __init__(self, 
                 options: SRBTrajoptOptions,
                 headless: bool = False) -> None:
        self.options = options
        self.plant = None
        self.scene_graph = None
        self.headless = headless
        self.meshcat = StartMeshcat() if not headless else None
        self.srb_diagram = None 

    def create_srb_diagram(self):
        """
        Sets up SRB plant definition, and creates diagram containing the
        SRB plant and scene graph

        Returns:
            srb_diagram: Diagram containing SRB plant and scene graph
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step = 0.001)
        if not self.headless:
            self.visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)
        box_size = self.options.dimensions
        box_color = self.options.color
        box_mass = self.options.mass 
        
        # create unit inertia for a box 
        box_unit_inertia = UnitInertia.SolidBox(box_size[0], box_size[1], box_size[2])
        
        # create Spatial Inertia
        box_spatial_inertia = SpatialInertia(mass = box_mass, p_PScm_E = [0, 0, 0], G_SP_E = box_unit_inertia)


        # Add SRB body to plant
        srb_instance = plant.AddModelInstance("body")
        body = plant.AddRigidBody("body", srb_instance, box_spatial_inertia)

        # register visual geometry with plant 
        shape = Box(box_size)
        plant.RegisterVisualGeometry(body, RigidTransform(), shape, "body_vis_geom", box_color)
        plant.Finalize()

        self.plant = plant
        self.srb_diagram = builder.Build()
        return self.srb_diagram


    def render_srb(self) -> None:
        """
        Visualizes the SRB in Drake Visualizer
        """
        if self.srb_diagram is None:
            raise ValueError("SRB diagram not initialized, call create_srb_diagram() first")
        
        self.meshcat.Delete()
        diagram_context = self.srb_diagram.CreateDefaultContext()
        plant_context = self.plant.GetMyContextFromRoot(diagram_context)
        X_WB = RigidTransform()
        X_WB.set_translation([0, 0, 1])
        self.plant.SetFreeBodyPose(plant_context, self.plant.GetBodyByName("body"), X_WB)
        # render the plant in Drake Visualizer
        self.srb_diagram.ForcedPublish(diagram_context)