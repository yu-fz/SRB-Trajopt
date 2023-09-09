import jax
import jax.numpy as jnp
import numpy as np
from .jax_utils import *

jax.config.update('jax_platform_name', 'cpu')
@jax.jit
def com_dircol_constraint_jit(
    h: float,
    com_k1: np.ndarray,
    com_dot_k1: np.ndarray,
    foot_forces_k1: np.ndarray,
    com_k2: np.ndarray,
    com_dot_k2: np.ndarray,
    mass: float, 
    gravity: np.ndarray):

    def eval_constraint(
            h : float,
            com_k1 : np.ndarray, 
            com_dot_k1 : np.ndarray,
            foot_forces_k1 : np.ndarray,
            com_k2 : np.ndarray,  
            com_dot_k2 : np.ndarray):
        
        sum_forces_k1 = jnp.sum(foot_forces_k1, axis=1)
        com_ddot_k1 = (sum_forces_k1 / mass) + gravity
        com_dot_kc = 0.5*(com_k1 + com_k2) + (h/8)*(com_ddot_k1 - com_dot_k2)#com_dot_k1 + (h/2)*com_ddot_k1
        # direct collocation constraint formula
        rhs = (-3/(2*h))*(com_k1 - com_k2) - (1/4)*(com_dot_k1 + com_dot_k2)
        return com_dot_kc - rhs 

    # Compute the value of the constraint
    constraint_val = eval_constraint(h, com_k1, com_dot_k1, foot_forces_k1, com_k2, com_dot_k2)

    # Compute the Jacobian using jacfwd
    jacobian_fn = jax.jacfwd(eval_constraint, argnums=tuple(range(6)))
    jacobian = jacobian_fn(h, com_k1, com_dot_k1, foot_forces_k1, com_k2, com_dot_k2)

    constraint_jac_jax_wrt_h = jacobian[0].reshape(3, -1,)
    constraint_jac_jax_wrt_com_k1 = jacobian[1]
    constraint_jac_jax_wrt_com_dot_k1 = jacobian[2]
    constraint_jac_jax_wrt_foot_forces_k1 = jacobian[3].reshape(3, -1, order='F')
    #print(constraint_jac_jax_wrt_foot_forces_k1.shape)
    constraint_jac_jax_wrt_com_k2 = jacobian[4]
    constraint_jac_jax_wrt_com_dot_k2 = jacobian[5]

    constraint_jac_jax_wrt_x = jnp.hstack((constraint_jac_jax_wrt_h, 
                                           constraint_jac_jax_wrt_com_k1,
                                           constraint_jac_jax_wrt_com_dot_k1,
                                           constraint_jac_jax_wrt_foot_forces_k1,
                                           constraint_jac_jax_wrt_com_k2,
                                           constraint_jac_jax_wrt_com_dot_k2))
    return constraint_val, constraint_jac_jax_wrt_x

@jax.jit
def com_dot_dircol_constraint_jit(
    h: float,
    com_dot_k1: np.ndarray,
    foot_forces_k1: np.ndarray,
    com_dot_k2: np.ndarray,
    foot_forces_k2: np.ndarray,
    mass: float,
    gravity: np.ndarray):

    def eval_constraint(
            h,
            com_dot_k1, 
            foot_forces_k1,
            com_dot_k2,  
            foot_forces_k2,):
        
        sum_forces_k1 = jnp.sum(foot_forces_k1, axis=1)
        sum_forces_k2 = jnp.sum(foot_forces_k2, axis=1)
        sum_forces_kc = (sum_forces_k1 + sum_forces_k2) / 2
        # normalize ground reaction forces
        com_ddot_k1 = gravity[2]*(sum_forces_k1 + jnp.array([0, 0, 1]))
        com_ddot_k2 = gravity[2]*(sum_forces_k2 + jnp.array([0, 0, 1]))
        com_ddot_kc = gravity[2]*(sum_forces_kc + jnp.array([0, 0, 1]))
        # direct collocation constraint formula
        rhs = (-3/(2*h))*(com_dot_k1 - com_dot_k2) - (1/4)*(com_ddot_k1 + com_ddot_k2)
        return com_ddot_kc - rhs 

    # Compute the value of the constraint
    constraint_val = eval_constraint(h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2)

    # Compute the Jacobian using jacrev
    jacobian_fn = jax.jacrev(eval_constraint, argnums=tuple(range(5)))
    jacobian = jacobian_fn(h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2)

    constraint_jac_jax_wrt_h = jacobian[0].reshape(3, -1,)
    constraint_jac_jax_wrt_com_dot_k1 = jacobian[1]
    constraint_jac_jax_wrt_foot_forces_k1 = jacobian[2].reshape(3, -1, order='F')
    #print(constraint_jac_jax_wrt_foot_forces_k1.shape)
    constraint_jac_jax_wrt_com_dot_k2 = jacobian[3]
    constraint_jac_jax_wrt_foot_forces_k2 = jacobian[4].reshape(3, -1, order='F')

    constraint_jac_jax_wrt_x = jnp.hstack((constraint_jac_jax_wrt_h, 
                                               constraint_jac_jax_wrt_com_dot_k1,
                                               constraint_jac_jax_wrt_foot_forces_k1,
                                               constraint_jac_jax_wrt_com_dot_k2,
                                               constraint_jac_jax_wrt_foot_forces_k2,))
    return constraint_val, constraint_jac_jax_wrt_x

@jax.jit 
def angvel_dircol_constraint_jit(
    h: float,
    k1_decision_vars: dict,
    k2_decision_vars: dict,
    foot_length: float,
    foot_width: float,
    I_BBo_B: np.ndarray,
    mass: float,
    ):

    def _get_foot_contact_positions(
        srb_yaw_rotation_mat,
        p_W_com,
        p_W_F,
        foot_length,
        foot_width
    ):
        """
        Computes the positions of the four contact points 
        of the left and right foot from the foot conctact location,
        and rotates the positions of the contact points to the body yaw angle d

        p_B_F_W: 3x1 array of the foot contact location measured from the body frame expressed in the world frame
        
        Returns:
            rotated_foot_contact_positions: 3x4 array of the rotated foot contact positions
        """
        p_B_F_W = p_W_F - p_W_com

        p_b_f_w_c1 = jnp.array([
            p_B_F_W[0] + foot_length/2,
            p_B_F_W[1] + foot_width/2,
            p_B_F_W[2]
        ]).reshape(3, 1)

        p_b_f_w_c2 = jnp.array([
            p_B_F_W[0] + foot_length/2,
            p_B_F_W[1] - foot_width/2,
            p_B_F_W[2]
        ]).reshape(3, 1)

        p_b_f_w_c3 = jnp.array([
            p_B_F_W[0] - foot_length/2,
            p_B_F_W[1] + foot_width/2,
            p_B_F_W[2]
        ]).reshape(3, 1)

        p_b_f_w_c4 = jnp.array([
            p_B_F_W[0] - foot_length/2,
            p_B_F_W[1] - foot_width/2,
            p_B_F_W[2]
        ]).reshape(3, 1)
        foot_contact_positions = jnp.hstack((p_b_f_w_c1, p_b_f_w_c2, p_b_f_w_c3, p_b_f_w_c4))
        rotated_foot_contact_positions = jnp.matmul(srb_yaw_rotation_mat, foot_contact_positions)
        return rotated_foot_contact_positions

    def _compute_omega_dot(
        inertia_mat,
        srb_yaw_rotation_mat,
        p_W_com,
        p_W_LF,
        p_W_RF,
        foot_forces,
        foot_torques,
    ):
        """
        computes body angular velocity dynamics omega_dot = f(x,u)
        as omega_dot = I^{-1}*(r_ixF_i + m_i)
        """
        left_foot_forces = foot_forces[:, 0:4]
        right_foot_forces = foot_forces[:, 4:8]
        #print(f"foot torques shape: {foot_torques.shape}")
        left_foot_torques = foot_torques[:, 0:4]
        right_foot_torques = foot_torques[:, 4:8]

        # left foot contact positions 
        p_B_LF_contacts = _get_foot_contact_positions(
            srb_yaw_rotation_mat,
            p_W_com,
            p_W_LF,
            foot_length,
            foot_width
        )
        # right foot contact positions
        p_B_RF_contacts = _get_foot_contact_positions(
            srb_yaw_rotation_mat,
            p_W_com,
            p_W_RF,
            foot_length,
            foot_width
        )

        # compute r x F for left and right foot
        r_x_F_LF = jnp.cross(p_B_LF_contacts, left_foot_forces, axis=0)
        r_x_F_RF = jnp.cross(p_B_RF_contacts, right_foot_forces, axis=0)
        sum_forces = jnp.sum(r_x_F_LF + r_x_F_RF, axis=1).reshape(3, 1)
        sum_torques = jnp.sum(left_foot_torques + right_foot_torques, axis=1).reshape(3, 1)
        sum_wrench = jnp.sum(sum_forces + sum_torques, axis=1)
        #sum_moments = jnp.sum(r_x_F_LF + r_x_F_RF + left_foot_torques + right_foot_torques, axis=1)
        # return omega dot 
        omega_dot = jnp.linalg.inv(inertia_mat)@sum_wrench
        return omega_dot

    def eval_constraint(
            h,
            com_k1,
            quat_k1,
            com_dot_k1,
            body_angvel_k1,
            p_W_LF_k1,
            p_W_RF_k1,
            foot_forces_k1,
            foot_torques_k1,
            com_k2,
            quat_k2,
            com_dot_k2,
            body_angvel_k2,
            p_W_LF_k2,
            p_W_RF_k2,
            foot_forces_k2,
            foot_torques_k2,
            ):
        
        # compute cont. dynamics at knot point 1
        # compute cont. dynamics at collocation point
        # compute cont. dynamics at knot point 2
        srb_rotation_euler_k1 = quaternion_to_euler(quat_k1)
        srb_yaw_rotation_mat_k1 = euler_to_SO3(0, 0, srb_rotation_euler_k1[2]) 
        
        I_B_W_k1 = calc_rotational_inertia_in_world_frame(
            quat_k1,
            com_k1,
            I_BBo_B,
            mass
        )

        I_B_W_k2 = calc_rotational_inertia_in_world_frame(
            quat_k2,
            com_k2,
            I_BBo_B,
            mass
        )

        omega_dot_k1 = _compute_omega_dot(
            I_B_W_k1,
            srb_yaw_rotation_mat_k1,
            com_k1,
            p_W_LF_k1,
            p_W_RF_k1,
            foot_forces_k1,
            foot_torques_k1,
        )

        foot_forces_kc = jnp.array([foot_forces_k1, foot_forces_k2]).mean(axis=0)
        foot_torques_kc = jnp.array([foot_torques_k1, foot_torques_k2]).mean(axis=0)

        # foot positions in stance should not change, so p_W_F_kc = p_W_F_k1 = p_W_F_k2
        p_W_LF_kc = jnp.array([p_W_LF_k1, p_W_LF_k2]).mean(axis=0)
        p_W_RF_kc = jnp.array([p_W_RF_k1, p_W_RF_k2]).mean(axis=0)
        # initial body orientation quat_k1 is used for all dynamics evaluations 
        # because the feet in contact do not move 
        # compute com_kc 
        com_kc = 0.5*(com_k1 + com_k2) + (h/8)*(com_dot_k1 - com_dot_k2)
        # quat_kc 
        quat_kc = 0.5*(quat_k1 + quat_k2) + (h/8)* \
            (calc_quaternion_derivative(quat_k1, body_angvel_k1) - \
             calc_quaternion_derivative(quat_k2, body_angvel_k2))
        I_B_W_kc = calc_rotational_inertia_in_world_frame(
            quat_kc,
            com_kc,
            I_BBo_B,
            mass
        )


        omega_dot_kc = _compute_omega_dot(
            I_B_W_kc,
            srb_yaw_rotation_mat_k1,
            com_kc,
            p_W_LF_kc,
            p_W_RF_kc,
            foot_forces_kc,
            foot_torques_kc,
        )

        omega_dot_k2 = _compute_omega_dot(
            I_B_W_k2,
            srb_yaw_rotation_mat_k1,
            com_k2,
            p_W_LF_k2,
            p_W_RF_k2,
            foot_forces_k2,
            foot_torques_k2,
        )

        # direct collocation constraint formula
        rhs = (-3/(2*h))*(body_angvel_k1 - body_angvel_k2) - (1/4)*(omega_dot_k1 + omega_dot_k2)
        return omega_dot_kc - rhs

    params_list = [
        h,
        k1_decision_vars['com_k1'],
        k1_decision_vars['quat_k1'],
        k1_decision_vars['com_dot_k1'],
        k1_decision_vars['body_angvel_k1'],
        k1_decision_vars['p_W_LF_k1'],
        k1_decision_vars['p_W_RF_k1'],
        k1_decision_vars['foot_forces_k1'],
        k1_decision_vars['foot_torques_k1'],
        k2_decision_vars['com_k2'],
        k2_decision_vars['quat_k2'],
        k2_decision_vars['com_dot_k2'],
        k2_decision_vars['body_angvel_k2'],
        k2_decision_vars['p_W_LF_k2'],
        k2_decision_vars['p_W_RF_k2'],
        k2_decision_vars['foot_forces_k2'],
        k2_decision_vars['foot_torques_k2'],
    ]

    # Compute the value of the constraint
    constraint_val = eval_constraint(
        *params_list
    ) 

    # Compute the Jacobian using jacrev for wide matrices
    jacobian_fn = jax.jacrev(eval_constraint, argnums=tuple(range(len(params_list))))
    #jacobian_fn = jax.jacrev(eval_constraint, argnums=(0,1))
    jacobian = jacobian_fn(
        *params_list
    )

    # Define a reshape function for use with vmap
    def reshape_jacobian(jac):
        if jac.shape == (3,):
            return jac.reshape(3, 1)
        elif jac.shape == (3, 3, 8):
            return jac.reshape(3, -1, order='F')
        else:
            return jac

    # Use vmap to reshape all Jacobians in the tuple
    reshaped_jacobians = tuple(map(reshape_jacobian, jacobian))

    # Stack the reshaped Jacobians horizontally
    constraint_jac_jax = jnp.hstack(reshaped_jacobians)
    # Assuming you have constraint_val defined
    return constraint_val, constraint_jac_jax

