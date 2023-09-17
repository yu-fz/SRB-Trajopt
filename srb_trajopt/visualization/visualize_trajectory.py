import time 
from util.colors import PASTEL_PEACH, PASTEL_AQUA

from pydrake.all import (
    Diagram,
    Rgba,
    RigidTransform,
    Context,
)
from pydrake.geometry import(
    Shape,
    Sphere,
    SceneGraph,
    GeometryFrame,
    GeometryInstance,
    FramePoseVector,
    SceneGraphInspector,
    QueryObject,
    SourceId,
    MakePhongIllustrationProperties
)
import numpy as np

def add_vis_object_to_scene(
    diagram: Diagram,
    geom_name: str,
    geom_shape: Shape,
    rgba: Rgba = Rgba(0.5, 0.5, 0.5, 1.0)
):
    """
    Adds a visualization geom to the scene graph of the given diagram 
    """
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    source_id = scene_graph.RegisterSource(geom_name)
    frame_id = scene_graph.RegisterFrame(source_id, GeometryFrame(geom_name))
    geom_instance = GeometryInstance(RigidTransform(),
                                     shape=geom_shape,
                                     name=geom_name,)
    illus_properties = MakePhongIllustrationProperties(rgba.rgba)
    geom_instance.set_illustration_properties(illus_properties)
    scene_graph.RegisterGeometry(source_id, frame_id, geom_instance)
    return source_id

def draw_vis_object_pose(
    scene_graph: SceneGraph,
    root_context: Context,
    source_id: SourceId,
    pose: RigidTransform
):
    """
    
    """
    scene_graph_context = scene_graph.GetMyContextFromRoot(root_context)
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    scene_graph_inspector =  query_object.inspector()
    geom_pose_port = scene_graph.get_source_pose_port(source_id)
    geom_frame_id = scene_graph_inspector.FramesForSource(source_id).pop()
    geom_pose = FramePoseVector()
    geom_pose.set_value(geom_frame_id, pose)
    geom_pose_port.FixValue(scene_graph_context, geom_pose)

def render_SRB_trajectory(
    srb_diagram : Diagram,
    trajectory_duration: float,
    trajectories,
):
    
    # Create visualization geometries for left and right foot
    left_foot_geom = Sphere(radius=0.025)
    left_foot_rgba = PASTEL_AQUA
    right_foot_geom = Sphere(radius=0.025)
    right_foot_rgba = PASTEL_PEACH
    left_foot_geom_src_id = add_vis_object_to_scene(diagram=srb_diagram, 
                                                geom_name="left_foot_geom", 
                                                geom_shape=left_foot_geom, 
                                                rgba=left_foot_rgba)
    
    right_foot_geom_src_id = add_vis_object_to_scene(diagram=srb_diagram, 
                                                 geom_name="right_foot_geom", 
                                                 geom_shape=right_foot_geom, 
                                                 rgba=right_foot_rgba)

    visualizer = srb_diagram.GetSubsystemByName("visualizer")
    scene_graph = srb_diagram.GetSubsystemByName("scene_graph")
    plant = srb_diagram.GetSubsystemByName("plant")
    context = srb_diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer.StartRecording()

    srb_floating_base_traj = trajectories[0]
    left_foot_pos_traj = trajectories[1]
    right_foot_pos_traj = trajectories[2]

    FPS = 30
    times = np.arange(start=0, 
                     stop=trajectory_duration, 
                     step=1/FPS)
    
    for i in range(len(times)):
        #TODO draw foot trajectories
        context.SetTime(times[i])
        srb_orientation = srb_floating_base_traj.get_orientation_trajectory().value(times[i])
        srb_com_pos = srb_floating_base_traj.get_position_trajectory().value(times[i])
        srb_state = np.concatenate((srb_orientation, srb_com_pos))
        p_W_LF = left_foot_pos_traj.value(times[i])
        p_W_RF = right_foot_pos_traj.value(times[i])
        # draw visualization markers for left and right foot positions
        draw_vis_object_pose(scene_graph=scene_graph,
                             root_context=context,
                             source_id=left_foot_geom_src_id,
                             pose=RigidTransform(p_W_LF))
        draw_vis_object_pose(scene_graph=scene_graph,
                             root_context=context,
                             source_id=right_foot_geom_src_id,
                             pose=RigidTransform(p_W_RF))

        plant.SetPositions(plant_context, srb_state)

        # update visualization markers for left and right foot positions
        srb_diagram.ForcedPublish(context)
    time.sleep(5)
    visualizer.StopRecording()
    visualizer.PublishRecording()

    
    # # Animate trajectory
    # context = diagram.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(context)
    # t_sol = np.cumsum(np.concatenate(([0], result.GetSolution(h))))
    # q_sol = PiecewisePolynomial.FirstOrderHold(t_sol, result.GetSolution(q))
    # visualizer.StartRecording()
    # num_strides = 4
    # t0 = t_sol[0]
    # tf = t_sol[-1]
    # T = tf * num_strides * (2.0 if is_laterally_symmetric else 1.0)
    # for t in np.hstack((np.arange(t0, T, 1.0 / 32.0), T)):
    #     context.SetTime(t)
    #     stride = (t - t0) // (tf - t0)
    #     ts = (t - t0) % (tf - t0)
    #     qt = PositionView(q_sol.value(ts))
    #     if is_laterally_symmetric:
    #         if stride % 2 == 1:
    #             qt = HalfStrideToFullStride(qt)
    #             qt._world_body_x += stride_length / 2.0
    #         stride = stride // 2
    #     qt._world_body_x += stride * stride_length
    #     plant.SetPositions(plant_context, qt[:])
    #     diagram.ForcedPublish(context)

    # visualizer.StopRecording()
    # visualizer.PublishRecording()
