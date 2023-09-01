from options import SRBTrajoptOptions
from pydrake.all import (
    PiecewisePolynomial,
    MathematicalProgram,
    MultibodyPlant,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    SpatialInertia,
    UnitInertia,
)

from pydrake.geometry import(
    Rgba,
)


class SRBTrajopt:
    def __init__(self, options: SRBTrajoptOptions) -> None:
        self.options = options
        self.plant = None
        self.scene_graph = None
        self.create_srb_diagram()

    def create_srb_diagram(self) -> MultibodyPlant:
        """
        Sets up SRB plant definition, and creates diagram containing the
        SRB plant and scene graph
        """
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step = 0.001)
        box_size = self.options.dimensions
        box_color = self.options.color
        box_mass = self.options.mass 
        
        # create unit inertia for a box 
        box_unit_inertia = UnitInertia.SolidBox(box_size[0], box_size[1], box_size[2])
        
        # create Spatial Inertia
        box_spatial_inertia = SpatialInertia(mass = box_mass, p_PScm_E = [0, 0, 0], G_SP_E = box_unit_inertia)

        # Add SRB body to plant
        srb_instance = plant.AddModelInstance("body")
        plant.AddRigidBody("body", srb_instance, box_spatial_inertia)
        plant.Finalize()
        body_index = plant.GetModelInstanceByName("body")
        print("body index: ", body_index)

