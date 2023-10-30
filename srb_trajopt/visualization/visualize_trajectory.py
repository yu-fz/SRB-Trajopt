import time 
from util.colors import *

from pydrake.all import (
    Diagram,
    Rgba,
    RigidTransform,
    Context,
    RollPitchYaw
)
from pydrake.geometry import(
    Shape,
    Sphere,
    Box,
    Capsule,
    SceneGraph,
    GeometryFrame,
    GeometryInstance,
    FramePoseVector,
    SceneGraphInspector,
    QueryObject,
    SourceId,
    Meshcat,
    MakePhongIllustrationProperties,
    Role,
)
import numpy as np

def register_leg_vis_sources(diagram: Diagram, geom_name: str, num_frames: int):
    scene_graph = diagram.GetSubsystemByName("scene_graph")
    source_ids = []
    for i in range(num_frames):
        source_id = scene_graph.RegisterSource(f"{geom_name}_{i}")
        source_ids.append(source_id)
    return source_ids

def add_vis_object_to_scene(
    diagram: Diagram,
    geom_name: str,
    geom_shape: Shape,
    rgba: Rgba = Rgba(0.5, 0.5, 0.5, 1.0),
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
    geom_id = scene_graph.RegisterGeometry(source_id, frame_id, geom_instance)

    return source_id, geom_id

# def add_vis_object_to_scene(
#     diagram: Diagram,
#     geom_name: str,
#     geom_shape: Shape,
#     rgba: Rgba = Rgba(0.5, 0.5, 0.5, 1.0),
#     context = None
# ):
#     """
#     Adds a visualization geom to the scene graph of the given diagram 
#     """
#     scene_graph = diagram.GetSubsystemByName("scene_graph")
#     source_id = scene_graph.RegisterSource(geom_name)
#     frame_id = scene_graph.RegisterFrame(source_id, GeometryFrame(geom_name))
#     geom_instance = GeometryInstance(RigidTransform(),
#                                      shape=geom_shape,
#                                      name=geom_name,)
#     illus_properties = MakePhongIllustrationProperties(rgba.rgba)
#     geom_instance.set_illustration_properties(illus_properties)
#     if context is not None:
#         print("context is not None")
#         scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
#         geom_id = scene_graph.RegisterGeometry(scene_graph_context, source_id, frame_id, geom_instance)
#         print(f"geom_id: {geom_id}")
#     else:
#         geom_id = scene_graph.RegisterGeometry(source_id, frame_id, geom_instance)
#     return source_id, geom_id 

def add_geometry_instance(
    scene_graph: SceneGraph,
    source_id: SourceId,
    shape: Shape,
    name: str,
    rgba: Rgba = Rgba(0.5, 0.5, 0.5, 1.0)
):
    scene_graph_inspector = scene_graph.model_inspector()
    frame_id = scene_graph_inspector.FramesForSource(source_id).pop()
    geom_instance = GeometryInstance(RigidTransform(),
                                     shape=shape,
                                     name=name,)
    illus_properties = MakePhongIllustrationProperties(rgba.rgba)
    geom_instance.set_illustration_properties(illus_properties)
    geom_id = scene_graph.RegisterGeometry(source_id, frame_id, geom_instance)
    return geom_id
                                     

def draw_vis_object_pose(
    scene_graph: SceneGraph,
    scene_graph_context: Context,
    source_id: SourceId,
    pose: RigidTransform
):
    """
    Updates the pose of the visualization geometry in the provided scene graph and context
    """
    query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
    scene_graph_inspector = query_object.inspector()
    geom_pose_port = scene_graph.get_source_pose_port(source_id)
    geom_frame_id = scene_graph_inspector.FramesForSource(source_id).pop()
    geom_pose = FramePoseVector()
    geom_pose.set_value(geom_frame_id, pose)
    geom_pose_port.FixValue(scene_graph_context, geom_pose)

def render_SRB_trajectory(
    srb_diagram : Diagram,
    meshcat : Meshcat,
    trajectory_duration: float,
    trajectories,
):
    """
    Animates the trajectory optimization solutions in meshcat
    """
    # Create visualization geometries for left and right foot
    left_foot_geom = Sphere(radius=0.025)
    left_foot_rgba = PASTEL_AQUA
    right_foot_geom = Sphere(radius=0.025)
    right_foot_rgba = PASTEL_PEACH
    left_leg_rgba = Rgba(0.0, 0.0, 0.0, 1.0)
    right_leg_rgba = Rgba(0.0, 0.0, 0.0, 1.0)
    leg_geom_radius = 0.01
 
    left_foot_vis_src_id, left_foot_vis_geom_id = add_vis_object_to_scene(
        diagram=srb_diagram, 
        geom_name="left_foot_geom", 
        geom_shape=left_foot_geom, 
        rgba=left_foot_rgba)
    
    right_foot_vis_src_id, right_foot_vis_geom_id = add_vis_object_to_scene(
        diagram=srb_diagram, 
        geom_name="right_foot_geom", 
        geom_shape=right_foot_geom, 
        rgba=right_foot_rgba)
        
    visualizer = srb_diagram.GetSubsystemByName("visualizer")
    scene_graph = srb_diagram.GetSubsystemByName("scene_graph")
    plant = srb_diagram.GetSubsystemByName("plant")
    # context = srb_diagram.CreateDefaultContext()
    # plant_context = plant.GetMyContextFromRoot(context)
    srb_floating_base_traj = trajectories[0]
    left_foot_pos_traj = trajectories[1]
    right_foot_pos_traj = trajectories[2]

    FPS = 30
    times = np.arange(start=0, 
                     stop=trajectory_duration, 
                     step=1/FPS)

    left_leg_vis_id_pairs = []
    right_leg_vis_id_pairs = []
    # context = srb_diagram.CreateDefaultContext()
    for i in range(len(times)):

        left_leg_vis_src_id, left_leg_vis_geom_id = add_vis_object_to_scene(
            diagram=srb_diagram, 
            geom_name=f"left_leg_geom/frame_{i}",
            geom_shape=Capsule(radius=leg_geom_radius, length=0.1),
            rgba=left_leg_rgba,
            )
        
        right_leg_vis_src_id, right_leg_vis_geom_id = add_vis_object_to_scene(
            diagram=srb_diagram,
            geom_name=f"right_leg_geom/frame_{i}",
            geom_shape=Capsule(radius=leg_geom_radius, length=0.1),
            rgba=right_leg_rgba,
            )
        
        # print(f"left_leg_vis_src_id: {left_leg_vis_src_id}")
        # print(f"right_leg_vis_src_id: {right_leg_vis_src_id}")
        # query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        # scene_graph_inspector = query_object.inspector()
        # geom_pose_port = scene_graph.get_source_pose_port(source_id)
        # geom_frame_id = scene_graph_inspector.FramesForSource(source_id).pop()
        # geom_pose = FramePoseVector()
        # geom_pose.set_value(geom_frame_id, pose)
        # geom_pose_port.FixValue(scene_graph_context, geom_pose)

        # scene_graph_context = scene_graph.GetMyContextFromRoot(context)
        # left_foot_geom_pose_port = scene_graph.get_source_pose_port(left_foot_vis_src_id)
        # right_foot_geom_pose_port = scene_graph.get_source_pose_port(right_foot_vis_src_id)
        # # left_foot_geom_pose_port.FixValue(scene_graph_context, FramePoseVector())
        # # right_foot_geom_pose_port.FixValue(scene_graph_context, FramePoseVector())


        # query_object = scene_graph.get_query_output_port().Eval(scene_graph_context)
        # scene_graph_inspector = query_object.inspector()

        # left_leg_geom_pose_port = scene_graph.get_source_pose_port(left_leg_vis_src_id)
        # # #right_leg_geom_pose_port = scene_graph.get_source_pose_port(right_leg_vis_src_id)
        
        # left_leg_frame_id = scene_graph_inspector.FramesForSource(left_leg_vis_src_id).pop()
        # left_foot_frame_id = scene_graph_inspector.FramesForSource(left_foot_vis_src_id).pop()
        # right_foot_frame_id = scene_graph_inspector.FramesForSource(right_foot_vis_src_id).pop()
        # # right_leg_frame_id = scene_graph_inspector.FramesForSource(right_leg_vis_src_id).pop()
        
        # default_pose_left_leg = FramePoseVector()
        # default_pose_left_foot = FramePoseVector()
        # default_pose_right = FramePoseVector()
        # default_pose_left_leg.set_value(left_leg_frame_id, RigidTransform())
        # default_pose_left_foot.set_value(left_foot_frame_id, RigidTransform())
        # default_pose_right.set_value(right_foot_frame_id, RigidTransform())
        # # default_pose_right.set_value(right_leg_frame_id, RigidTransform())
        # left_leg_geom_pose_port.FixValue(scene_graph_context, default_pose_left_leg)
        # left_foot_geom_pose_port.FixValue(scene_graph_context, default_pose_left_foot)
        # right_foot_geom_pose_port.FixValue(scene_graph_context, default_pose_right)
        # scene_graph.ForcedPublish(scene_graph_context)
        # right_leg_geom_pose_port.FixValue(scene_graph_context, default_pose_right)
        # scene_graph.ForcedPublish(scene_graph_context)
        # srb_diagram.ForcedPublish(context)
        left_leg_vis_id_pairs.append((left_leg_vis_src_id, left_leg_vis_geom_id))
        right_leg_vis_id_pairs.append((right_leg_vis_src_id, right_leg_vis_geom_id))
    
    context = srb_diagram.CreateDefaultContext()
    def compute_leg_ik(p_W_CoM, p_W_F):
        """
        Return the RigidTransform of the leg so that it points to the position of 
        the foot from the body CoM frame
        """
        # 1. compute the Z axis distance between the CoM and the foot 
        delta_z = p_W_F[2] - p_W_CoM[2]
        
        # 2. compute the X axis distance between the CoM and the foot
        delta_x = p_W_F[0] - p_W_CoM[0]
        # 3. compute the Y axis distance between the CoM and the foot
        delta_y = p_W_F[1] - p_W_CoM[1]
        # compute the angle of the leg in the XZ plane. This is the leg pitch angle
        theta_pitch = np.arctan2(delta_x, delta_z).item()
        # compute the angle of the leg in the YZ plane. This is the leg roll angle
        theta_roll = np.arctan2(delta_y, delta_z).item()
        # Create RollPitchYaw object from the computed angles
        rpy = RollPitchYaw(theta_roll, theta_pitch, 0.)
        # return RigidTransform result 
        return RigidTransform(rpy, p_W_CoM)

    for i in range(len(times)):
        context.SetTime(times[i])
        scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
        srb_com_pos = srb_floating_base_traj.get_position_trajectory().value(times[i])
        p_W_LF = left_foot_pos_traj.value(times[i])
        p_W_RF = right_foot_pos_traj.value(times[i])
        # left_leg_vis_src_id, left_leg_vis_geom_id = add_vis_object_to_scene(
        #     diagram=srb_diagram, 
        #     geom_name=f"left_leg_geom/frame_{i}",
        #     geom_shape=Capsule(radius=leg_geom_radius, length=0.1),
        #     rgba=left_leg_rgba,
        #     context=context)
        
        # right_leg_vis_src_id, right_leg_vis_geom_id = add_vis_object_to_scene(
        #     diagram=srb_diagram,
        #     geom_name=f"right_leg_geom/frame_{i}",
        #     geom_shape=Capsule(radius=leg_geom_radius, length=0.1),
        #     rgba=right_leg_rgba,
        #     context=context)
        # print(f"left_leg_vis_src_id: {left_leg_vis_src_id}")
        # # srb_diagram.ForcedPublish(context)
        # print(left_leg_vis_src_id, left_leg_vis_geom_id)
        # print(right_leg_vis_src_id, right_leg_vis_geom_id)

        X_W_LL = compute_leg_ik(srb_com_pos, p_W_LF)
        X_W_RL = compute_leg_ik(srb_com_pos, p_W_RF)
        left_leg_len = np.linalg.norm(p_W_LF - srb_com_pos)
        right_leg_len = np.linalg.norm(p_W_RF - srb_com_pos)
        leg_leg_frame_offset_z = -left_leg_len / 2
        right_leg_frame_offset_z = -right_leg_len / 2
        X_B_Bp_ll = RigidTransform([0, 0, leg_leg_frame_offset_z])
        X_B_Bp_rl = RigidTransform([0, 0, right_leg_frame_offset_z])
        X_W_LL = X_W_LL.multiply(X_B_Bp_ll)
        X_W_RL = X_W_RL.multiply(X_B_Bp_rl)

        scene_graph.ChangeShape(
            context=scene_graph_context,
            source_id=left_leg_vis_id_pairs[i][0],
            geometry_id=left_leg_vis_id_pairs[i][1],
            shape=Capsule(leg_geom_radius, length=left_leg_len),
        )
        scene_graph.ChangeShape(
            context=scene_graph_context,
            source_id=right_leg_vis_id_pairs[i][0],
            geometry_id=right_leg_vis_id_pairs[i][1],
            shape=Capsule(radius=leg_geom_radius, length=right_leg_len),
        )
        draw_vis_object_pose(scene_graph=scene_graph,
                             scene_graph_context=scene_graph_context,
                             source_id=left_leg_vis_id_pairs[i][0],
                             pose=X_W_LL)
        draw_vis_object_pose(scene_graph=scene_graph,
                             scene_graph_context=scene_graph_context,
                             source_id=right_leg_vis_id_pairs[i][0],
                             pose=X_W_RL)
        scene_graph.ForcedPublish(scene_graph_context)
        # print(f"publish frame {i}")
        # vis_context = visualizer.GetMyContextFromRoot(context)
        # visualizer.ForcedPublish(vis_context)
        #srb_diagram.ForcedPublish(context)
        # meshcat.SetLineSegments(
        #     path=f"srb_vis_elements/frame_{i}/left_leg",
        #     start=srb_com_pos,
        #     end=p_W_LF,
        #     line_width=0.01,
        #     rgba=left_leg_rgba,
        # )
        # meshcat.SetLineSegments(
        #     path=f"srb_vis_elements/frame_{i}/right_leg",
        #     start=srb_com_pos,
        #     end=p_W_RF,
        #     line_width=0.01,
        #     rgba=right_leg_rgba,
        # )
    # time.sleep(5)
    # visualizer.StopRecording()
    # visualizer.PublishRecording()
    visualizer.StartRecording(True)
    # meshcat.StartRecording(set_visualizations_while_recording=True)
    for i in range(len(times)):
        plant_context = plant.GetMyContextFromRoot(context)
        context.SetTime(times[i])
        srb_orientation = srb_floating_base_traj.get_orientation_trajectory().value(times[i])
        srb_com_pos = srb_floating_base_traj.get_position_trajectory().value(times[i])
        srb_state = np.concatenate((srb_orientation, srb_com_pos))
        p_W_LF = left_foot_pos_traj.value(times[i])
        p_W_RF = right_foot_pos_traj.value(times[i])

        scene_graph_context = scene_graph.GetMyContextFromRoot(context)

        plant.SetPositions(
            context=plant_context, 
            model_instance = plant.GetModelInstanceByName("body"),
            q = srb_state)
        
        # draw visualization markers for left and right foot positions
        # scene_graph.ForcedPublish(scene_graph_context)
        draw_vis_object_pose(scene_graph=scene_graph,
                             scene_graph_context=scene_graph_context,
                             source_id=left_foot_vis_src_id,
                             pose=RigidTransform(p_W_LF))
        draw_vis_object_pose(scene_graph=scene_graph,
                             scene_graph_context=scene_graph_context,
                             source_id=right_foot_vis_src_id,
                             pose=RigidTransform(p_W_RF))
        
        for j in range(len(times)):
            if j != i:
                # set visibility of the previous leg visualization element to false
                meshcat.SetProperty(
                    path=f"/drake/visualizer/left_leg_geom/frame_{j}",
                    property="visible",
                    value=False,
                    time_in_recording=times[i],
                )
                meshcat.SetProperty(
                    path=f"/drake/visualizer/right_leg_geom/frame_{j}",
                    property="visible",
                    value=False,
                    time_in_recording=times[i],
                )
            else:
                meshcat.SetProperty(
                    path=f"/drake/visualizer/left_leg_geom/frame_{j}",
                    property="visible",
                    value=True,
                    time_in_recording=times[i],
                )
                meshcat.SetProperty(
                    path=f"/drake/visualizer/right_leg_geom/frame_{j}",
                    property="visible",
                    value=True,
                    time_in_recording=times[i],
                )
            #     scene_graph.RemoveRole(
            #         context=scene_graph_context,
            #         source_id=left_leg_vis_id_pairs[j][0],
            #         geometry_id=left_leg_vis_id_pairs[j][1],
            #         role=Role.kIllustration
            #     )
            #     scene_graph.RemoveRole(
            #         context=scene_graph_context,
            #         source_id=right_leg_vis_id_pairs[j][0],
            #         geometry_id=right_leg_vis_id_pairs[j][1],
            #         role=Role.kIllustration
            #     )
            #     scene_graph.AssignRole(
            #         context=scene_graph_context,
            #         source_id=left_leg_vis_id_pairs[j][0],
            #         geometry_id=left_leg_vis_id_pairs[j][1],
            #         properties=MakePhongIllustrationProperties(np.zeros(4))
            #     )
            #     scene_graph.AssignRole(
            #         context=scene_graph_context,
            #         source_id=right_leg_vis_id_pairs[j][0],
            #         geometry_id=right_leg_vis_id_pairs[j][1],
            #         properties=MakePhongIllustrationProperties(np.zeros(4))
            #     )
            # else:
            #     scene_graph.RemoveRole(
            #         context=scene_graph_context,
            #         source_id=left_leg_vis_id_pairs[j][0],
            #         geometry_id=left_leg_vis_id_pairs[j][1],
            #         role=Role.kIllustration
            #     )
            #     scene_graph.RemoveRole(
            #         context=scene_graph_context,
            #         source_id=right_leg_vis_id_pairs[j][0],
            #         geometry_id=right_leg_vis_id_pairs[j][1],
            #         role=Role.kIllustration
            #     )
            #     scene_graph.AssignRole(
            #         context=scene_graph_context,
            #         source_id=left_leg_vis_id_pairs[j][0],
            #         geometry_id=left_leg_vis_id_pairs[j][1],
            #         properties=MakePhongIllustrationProperties(np.ones(4))
            #     )
            #     scene_graph.AssignRole(
            #         context=scene_graph_context,
            #         source_id=right_leg_vis_id_pairs[j][0],
            #         geometry_id=right_leg_vis_id_pairs[j][1],
            #         properties=MakePhongIllustrationProperties(np.ones(4))
            #     )


        # for j in range(len(times)):
        # draw_vis_object_pose(scene_graph=scene_graph,
        #                     scene_graph_context=scene_graph_context,
        #                     source_id=left_leg_vis_id_pairs[i][0],
        #                     pose=X_W_LL)
            # if j == i:
            #     draw_vis_object_pose(scene_graph=scene_graph,
            #                         scene_graph_context=scene_graph_context,
            #                         source_id=left_leg_vis_id_pairs[j][0],
            #                         pose=X_W_LL)
            # else:
            #     draw_vis_object_pose(scene_graph=scene_graph,
            #                          scene_graph_context=scene_graph_context,
            #                          source_id=left_leg_vis_id_pairs[j][0],
            #                          pose=RigidTransform())
        # draw_vis_object_pose(scene_graph=scene_graph,
        #                      scene_graph_context=scene_graph_context,
        #                      source_id=right_leg_vis_id_pairs[i][0],
        #                      pose=X_W_RL)
        scene_graph.ForcedPublish(scene_graph_context)
        context = srb_diagram.GetMyContextFromRoot(context)
        srb_diagram.ForcedPublish(context)
        # # scene_graph_context = scene_graph.GetMyMutableContextFromRoot(context)
        # # scene_graph.ForcedPublish(scene_graph_context)
        # vis_context = visualizer.GetMyContextFromRoot(context)
        # visualizer.ForcedPublish(vis_context)
    time.sleep(4)
    # meshcat.StopRecording()
    # meshcat.PublishRecording()
    visualizer.StopRecording()
    visualizer.PublishRecording()
