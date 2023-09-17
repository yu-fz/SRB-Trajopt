from srb_trajopt.options import SRBTrajoptOptions
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
        self.headless = headless
        self.meshcat = StartMeshcat() if not headless else None

    def create_srb_diagram(self):
        """
        Sets up SRB plant definition, registers visual geometries, and returns the containing diagram

        Returns:
            srb_diagram: Diagram containing SRB plant, scene graph, and visualizer
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step = 0)
        if not self.headless:
            meshcat_vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)
            meshcat_vis.set_name("visualizer")
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

        srb_diagram = builder.Build()
        ad_srb_diagram = srb_diagram.ToAutoDiffXd()
        return srb_diagram, ad_srb_diagram
