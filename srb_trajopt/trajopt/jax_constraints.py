import jax
import jax.numpy as jnp
import numpy as np
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
        com_dot_kc = com_dot_k1 + (h/2)*com_ddot_k1
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

    flattened_constraint_jac_jax = jnp.hstack((constraint_jac_jax_wrt_h, 
                                               constraint_jac_jax_wrt_com_k1,
                                               constraint_jac_jax_wrt_com_dot_k1,
                                               constraint_jac_jax_wrt_foot_forces_k1,
                                               constraint_jac_jax_wrt_com_k2,
                                               constraint_jac_jax_wrt_com_dot_k2))
    return constraint_val, flattened_constraint_jac_jax


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
        sum_forces_k_c = (sum_forces_k1 + sum_forces_k2) / 2
        com_ddot_k1 = (sum_forces_k1 / mass) + gravity
        com_ddot_k2 = (sum_forces_k2 / mass) + gravity
        com_ddot_kc = (sum_forces_k_c / mass) + gravity
        # direct collocation constraint formula
        rhs = (-3/(2*h))*(com_dot_k1 - com_dot_k2) - (1/4)*(com_ddot_k1 + com_ddot_k2)
        return com_ddot_kc - rhs 

    # Compute the value of the constraint
    constraint_val = eval_constraint(h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2)

    # Compute the Jacobian using jacfwd
    jacobian_fn = jax.jacfwd(eval_constraint, argnums=tuple(range(5)))
    jacobian = jacobian_fn(h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2)

    constraint_jac_jax_wrt_h = jacobian[0].reshape(3, -1,)
    constraint_jac_jax_wrt_com_dot_k1 = jacobian[1]
    constraint_jac_jax_wrt_foot_forces_k1 = jacobian[2].reshape(3, -1, order='F')
    #print(constraint_jac_jax_wrt_foot_forces_k1.shape)
    constraint_jac_jax_wrt_com_dot_k2 = jacobian[3]
    constraint_jac_jax_wrt_foot_forces_k2 = jacobian[4].reshape(3, -1, order='F')

    flattened_constraint_jac_jax = jnp.hstack((constraint_jac_jax_wrt_h, 
                                               constraint_jac_jax_wrt_com_dot_k1,
                                               constraint_jac_jax_wrt_foot_forces_k1,
                                               constraint_jac_jax_wrt_com_dot_k2,
                                               constraint_jac_jax_wrt_foot_forces_k2,))
    return constraint_val, flattened_constraint_jac_jax


@jax.jit
def angvel_dircol_constraint_jit(
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
        sum_forces_k_c = (sum_forces_k1 + sum_forces_k2) / 2
        com_ddot_k1 = (sum_forces_k1 / mass) + gravity
        com_ddot_k2 = (sum_forces_k2 / mass) + gravity
        com_ddot_kc = (sum_forces_k_c / mass) + gravity
        # direct collocation constraint formula
        rhs = (-3/(2*h))*(com_dot_k1 - com_dot_k2) - (1/4)*(com_ddot_k1 + com_ddot_k2)
        return com_ddot_kc - rhs 

    # Compute the value of the constraint
    constraint_val = eval_constraint(h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2)

    # Compute the Jacobian using jacfwd
    jacobian_fn = jax.jacfwd(eval_constraint, argnums=tuple(range(5)))
    jacobian = jacobian_fn(h, com_dot_k1, foot_forces_k1, com_dot_k2, foot_forces_k2)

    constraint_jac_jax_wrt_h = jacobian[0].reshape(3, -1,)
    constraint_jac_jax_wrt_com_dot_k1 = jacobian[1]
    constraint_jac_jax_wrt_foot_forces_k1 = jacobian[2].reshape(3, -1, order='F')
    #print(constraint_jac_jax_wrt_foot_forces_k1.shape)
    constraint_jac_jax_wrt_com_dot_k2 = jacobian[3]
    constraint_jac_jax_wrt_foot_forces_k2 = jacobian[4].reshape(3, -1, order='F')

    flattened_constraint_jac_jax = jnp.hstack((constraint_jac_jax_wrt_h, 
                                               constraint_jac_jax_wrt_com_dot_k1,
                                               constraint_jac_jax_wrt_foot_forces_k1,
                                               constraint_jac_jax_wrt_com_dot_k2,
                                               constraint_jac_jax_wrt_foot_forces_k2,))
    return constraint_val, flattened_constraint_jac_jax
