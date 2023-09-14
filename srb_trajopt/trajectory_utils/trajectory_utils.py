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

    # turn quaternion samples into PiecewiseQuaternionSlerp Trajectory
    drake_quat_srb_quat_soln = [Quaternion(wxyz=q) for q in srb_body_quat_soln.T]
    quat_trajectory = PiecewiseQuaternionSlerp(
        breaks=timesteps_soln,
        quaternions=drake_quat_srb_quat_soln,
    )

    com_pos_trajectory = PiecewisePolynomial.CubicHermite(
        breaks=timesteps_soln,
        samples=srb_body_com_pos_soln,
        samples_dot=srb_body_com_dot_soln
    )

    # combine srb position and orientation trajectories into a single PieceWisePose trajectory 
    srb_floating_base_trajectory = PiecewisePose(
        position_trajectory=com_pos_trajectory,
        orientation_trajectory=quat_trajectory,
    )

    left_foot_pos_trajectory = PiecewisePolynomial.FirstOrderHold(
        breaks=timesteps_soln,
        samples=p_W_LF_soln,
    )

    right_foot_pos_trajectory = PiecewisePolynomial.FirstOrderHold(
        breaks=timesteps_soln,
        samples=p_W_RF_soln,
    )

    return (srb_floating_base_trajectory, left_foot_pos_trajectory, right_foot_pos_trajectory)