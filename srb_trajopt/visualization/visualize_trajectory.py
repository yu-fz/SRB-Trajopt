
from pydrake.all import (
    Diagram,
)
import numpy as np
def render_SRB_trajectory(
    srb_diagram : Diagram,
    trajectory_duration: float,
    trajectories,
):
    visualizer = srb_diagram.GetSubsystemByName("visualizer")
    plant = srb_diagram.GetSubsystemByName("plant")
    context = srb_diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    visualizer.StartRecording()

    srb_floating_base_traj = trajectories[0]
    left_foot_pos_traj = trajectories[1]
    right_foot_pos_traj = trajectories[2]

    FPS = 30
    time = np.arange(start=0, 
                     stop=trajectory_duration, 
                     step=1/FPS)
    
    for i in range(len(time)):
        #TODO draw foot trajectories
        context.SetTime(time[i])
        plant.SetPositions(plant_context, srb_floating_base_traj.value(time[i]))
        srb_diagram.ForcedPublish(context)

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
