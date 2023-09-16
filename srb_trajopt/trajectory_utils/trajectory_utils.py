import numpy as np
from pydrake.trajectories import PiecewisePolynomial, PiecewiseQuaternionSlerp, PiecewisePose
from pydrake.all import Quaternion
def make_solution_trajectory(
        timesteps_soln: np.ndarray,
        srb_body_quat_soln: np.ndarray,
        srb_body_com_pos_soln: np.ndarray,
        srb_body_com_dot_soln: np.ndarray,
        p_W_LF_soln: np.ndarray,
        p_W_RF_soln: np.ndarray,):
    """
    Turns the discrete decisision variable solution vectors to a
    continuous trajectory
    """
    print(f"timesteps: {timesteps_soln}")
    # turn quaternion samples into PiecewiseQuaternionSlerp Trajectory
    srb_quat_soln_drake = []
    for idx in range(srb_body_quat_soln.shape[1]):
        quat = srb_body_quat_soln[:, idx]
        if np.linalg.norm(quat) != 1.0:
            quat = quat / np.linalg.norm(quat)
        q = Quaternion(wxyz=quat)
        srb_quat_soln_drake.append(q)

    #drake_quat_srb_quat_soln = [Quaternion(wxyz=q) for q in srb_body_quat_soln.T]
    quat_trajectory = PiecewiseQuaternionSlerp(
        breaks=timesteps_soln,
        quaternions=srb_quat_soln_drake,
    )

    com_pos_trajectory = PiecewisePolynomial.CubicHermite(
        breaks=timesteps_soln,
        samples=srb_body_com_pos_soln,
        samples_dot=srb_body_com_dot_soln
    )
    print(com_pos_trajectory.value(0.1))
    print(quat_trajectory.value(0.1))

    # combine srb position and orientation trajectories into a single PieceWisePose trajectory 
    srb_floating_base_trajectory = PiecewisePose(
        position_trajectory=com_pos_trajectory,
        orientation_trajectory=quat_trajectory,
    )
    print(srb_floating_base_trajectory.value(0.1))

    left_foot_pos_trajectory = PiecewisePolynomial.FirstOrderHold(
        breaks=timesteps_soln,
        samples=p_W_LF_soln,
    )

    right_foot_pos_trajectory = PiecewisePolynomial.FirstOrderHold(
        breaks=timesteps_soln,
        samples=p_W_RF_soln,
    )

    return (srb_floating_base_trajectory, left_foot_pos_trajectory, right_foot_pos_trajectory)