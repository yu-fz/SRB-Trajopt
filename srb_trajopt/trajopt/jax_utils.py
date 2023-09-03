import jax
import jax.numpy as jnp
import numpy as np


@jax.jit
def np_to_jax(np_array):
    return jnp.array(np_array)

# Convert multiple NumPy arrays to JAX arrays in one go
@jax.jit
def batch_np_to_jax(*np_arrays):
    return [jnp.array(np_array, copy=False) for np_array in np_arrays]