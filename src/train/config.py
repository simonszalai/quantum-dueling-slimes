"""
This file stores config variables that are expected to change less frequently.
"""

import numpy as np
import tensorflow as tf

# Dimensions for state and actions (depends on environment)
state_dim = 12
# Possible actions for slimevolley:
#   0: nothing
#   1: forward
#   2: jump
#   3: backward
#   4: forward + jump
#   5: backward + jump
action_dim = 6

# Dimension of the quantum noise added to the input of the habitual network
noise_dim = 8

# Parameters used to compute precision factor Ω.
# In cognitive terms, it can be taught of as top-down attention. Effect is to incentivize disentanglement of latent state.
# a: The sum a+d show the maximum value of omega
# b: This shows the average value of D_kl[pi] that will cause half sigmoid (i.e. d+a/2)
# c: This moves the steepness of the sigmoid
# d: This is the minimum omega (when sigmoid is zero)
omega_params = {"a": 1.0, "b": 5.0, "c": 15.0, "d": 0.1}


# Weight for state / observation when calculating loss for state encoder network. Seems to be unused.
beta_state = 1.0
beta_obs = 1.0

# γ regularization hyperparameter. Speeds up convergence, but doesn't influence behavior of the agent.
gamma = 0.0
gamma_rate = 0.01
gamma_max = 0.8
gamma_delay = 30

# Defines how many steps in the future G should be calculated
calc_G_steps_ahead = 2

# Defines how many samples should be averaged to calculate G
average_G_over_N_samples = 2

# Set data type to be used for matrices
np_precision = np.float32
tf_precision = tf.float32

# Learning rates for the 3 neural networks
learning_rates = {"habitual": 1e-04, "transition": 1e-04, "encoder": 0.001}
