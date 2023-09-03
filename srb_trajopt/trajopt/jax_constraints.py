import jax
import jax.numpy as jnp
import numpy as np
jax.config.update('jax_platform_name', 'cpu')

@jax.jit
def com_ddot_constraint_jit(
    com_ddot: np.ndarray,
    foot_forces: np.ndarray, 
    mass: float,
    gravity: np.ndarray):
    def eval_constraint(com_ddot, foot_forces):
        sum_forces = jnp.sum(foot_forces, axis=1)
        mg = mass * gravity[2]
        acceleration = jnp.array([0., 0., 1.]) + (sum_forces / mg)
        return acceleration - com_ddot / gravity[2]

    # Compute the value of the constraint
    constraint_val = eval_constraint(com_ddot, foot_forces)

    # Compute the Jacobian using jacfwd
    jacobian_fn = jax.jacfwd(eval_constraint, argnums=(0, 1,))
    jacobian = jacobian_fn(com_ddot, foot_forces)

    constraint_jac_jax_wrt_com_ddot = jacobian[0]
    constraint_jac_jax_wrt_foot_forces = jacobian[1].reshape(3, -1,)
    flattened_constraint_jac_jax = jnp.hstack((constraint_jac_jax_wrt_com_ddot, constraint_jac_jax_wrt_foot_forces))

    return constraint_val, flattened_constraint_jac_jax
