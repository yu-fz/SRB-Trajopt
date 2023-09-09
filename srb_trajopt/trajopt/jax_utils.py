import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import inspect

# Function to determine the number of arguments dynamically and cache the result
def get_num_args(fn):
    if hasattr(fn, '_num_args'):
        return fn._num_args
    num_args = len(inspect.signature(fn).parameters)
    fn._num_args = num_args  # Cache the result
    return num_args


@jax.jit
def np_to_jax(np_array):
    return jnp.array(np_array)

# Convert multiple NumPy arrays to JAX arrays in one go
@jax.jit
def batch_np_to_jax(*np_arrays):
    return [jnp.array(np_array, copy=False) for np_array in np_arrays]

@jax.jit
def quaternion_to_SO3(quat: jnp.array):
    """
    converts a w,x,y,z quaternion to a 3x3 rotation matrix
    """
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]
    return jnp.array([[1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]])

@partial(jax.jit, static_argnums=0)
def quaternion_to_euler(quat: jnp.array):
    """
    converts a w,x,y,z quaternion to extrinsinc roll, pitch, yaw euler angles
    Uses Drake's conversion algorithm found in 
    https://github.com/RobotLocomotion/drake/blob/362962b98a38fbadfa3bbe3a6d1791850de70f9a/math/roll_pitch_yaw.cc#L111
    """
    rot_mat = quaternion_to_SO3(quat)
    
    # use the high accuracy algorithm used in drake to convert 
    # quaternions to RPY euler angles

    R22 = rot_mat[2, 2]
    R21 = rot_mat[2, 1]
    R10 = rot_mat[1, 0]
    R00 = rot_mat[0, 0]
    Rsum = jnp.sqrt((R22 * R22 + R21 * R21 + R10 * R10 + R00 * R00) / 2)
    R20 = rot_mat[2, 0]
    q2 = jnp.arctan2(-R20, Rsum)

    e0 = quat[0]
    e1 = quat[1]
    e2 = quat[2]
    e3 = quat[3]
    yA = e1 + e3
    xA = e0 - e2
    yB = e3 - e1
    xB = e0 + e2
    epsilon = 1e-7
    isSingularA = jnp.logical_and(jnp.abs(yA) <= epsilon, jnp.abs(xA) <= epsilon)
    isSingularB = jnp.logical_and(jnp.abs(yB) <= epsilon, jnp.abs(xB) <= epsilon)
    zA = jnp.where(isSingularA, 0.0, jnp.arctan2(yA, xA))
    zB = jnp.where(isSingularB, 0.0, jnp.arctan2(yB, xB))
    q1 = zA - zB
    q3 = zA + zB

    # With this code block
    q1 = jnp.where(q1 > jnp.pi, q1 - 2 * jnp.pi, q1)
    q1 = jnp.where(q1 < -jnp.pi, q1 + 2 * jnp.pi, q1)
    q3 = jnp.where(q3 > jnp.pi, q3 - 2 * jnp.pi, q3)
    q3 = jnp.where(q3 < -jnp.pi, q3 + 2 * jnp.pi, q3)

    return jnp.array([q1, q2, q3])

@jax.jit
def euler_to_SO3(x: float, y: float, z: float):
    """
    Converts extrinsic roll, pitch, yaw euler angles to a 3x3 rotation matrix.
    """
    SO3 = jnp.array([
        [jnp.cos(z) * jnp.cos(y), jnp.cos(z) * jnp.sin(y) * jnp.sin(x) - jnp.sin(z) * jnp.cos(x),
         jnp.cos(z) * jnp.sin(y) * jnp.cos(x) + jnp.sin(z) * jnp.sin(x)],
        [jnp.sin(z) * jnp.cos(y), jnp.sin(z) * jnp.sin(y) * jnp.sin(x) + jnp.cos(z) * jnp.cos(x),
         jnp.sin(z) * jnp.sin(y) * jnp.cos(x) - jnp.cos(z) * jnp.sin(x)],
        [-jnp.sin(y), jnp.cos(y) * jnp.sin(x), jnp.cos(y) * jnp.cos(x)]
    ])
    return SO3

@jax.jit
def get_skew_symmetric_matrix(v: jnp.array):
    """
    Returns the skew symmetric matrix of a 3x1 vector.
    """
    return jnp.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])
@jax.jit
def calc_Omega_function(w: jnp.array):
    """
    Computes the Omega operator for the 3x1 angular velocity vector w.
    Used to compute the quaternion time derivative.
    Algorithm taken from:
    https://ahrs.readthedocs.io/en/latest/filters/angular.html
    """

    omega_skew = get_skew_symmetric_matrix(w)

    Omega = jnp.zeros((4, 4))
    Omega = Omega.at[1:4, 1:4].set(-omega_skew)
    Omega = Omega.at[0, 1:4].set(-w.T)
    Omega = Omega.at[1:4, 0].set(w)

    return Omega

@jax.jit
def calc_quaternion_derivative(quat: jnp.array, omega: jnp.array):
    """
    Calculates the derivative of a quaternion given a quaternion and an angular velocity.
    """
    quat_dot = 0.5 * calc_Omega_function(omega) @ quat
    return quat_dot


@jax.jit
def calc_rotational_inertia_in_world_frame(quat: jnp.array,
                                           p_W_CoM: jnp.array, 
                                           I_BBo_B: jnp.array,
                                           mass: float):
    """
    Calculates the rotational inertia in the world frame
    given a constant body frame rotational inertia and the position + orientation of the body
    """
    R = quaternion_to_SO3(quat)
    return R @ I_BBo_B @ R.T + mass * get_skew_symmetric_matrix(p_W_CoM) @ get_skew_symmetric_matrix(p_W_CoM).T