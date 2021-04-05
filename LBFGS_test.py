import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# A high-dimensional quadratic bowl.
ndims = 60
minimum = tf.ones([ndims], dtype='float64')
scales = tf.range(ndims, dtype='float64') + 1.0

# The objective function and the gradient.
@tf.function
def quadratic(x):
    value = tf.reduce_sum(scales * (x - minimum) ** 2)
    return value, tf.gradients(value, x)[0]

start = tf.range(ndims, 0, -1, dtype='float64')
optim_results = tfp.optimizer.lbfgs_minimize(
    quadratic, initial_position=start, num_correction_pairs=10,
    tolerance=1e-8)

# Check that the search converged
print(optim_results.converged.numpy())
# True

# Check that the argmin is close to the actual value.
print(np.allclose(optim_results.position.numpy(), minimum.numpy()))
# True