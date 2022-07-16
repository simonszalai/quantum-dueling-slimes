import numpy as np
import tensorflow as tf
from src.train.config import np_precision

eps = np.finfo(np_precision).eps * 10


def stable_tf_log(x):
    return tf.math.log(x + eps)


def stable_np_log(x):
    return np.log(x + eps)


def D_KL_from_logvar_and_precision(mu1, logvar1, mu2, logvar2, omega):
    D_KL = 0.5 * (logvar2 - stable_tf_log(omega) - logvar1) + (tf.exp(logvar1) + tf.math.square(mu1 - mu2)) / (2.0 * tf.exp(logvar2) / omega) - 0.5
    return tf.reduce_sum(D_KL, 1)


def entropy_bernoulli(p):
    return -(1 - p) * stable_tf_log(1 - p) - p * stable_tf_log(p)


def log_bernoulli(x, p):
    return x * stable_tf_log(p) + (1 - x) * stable_tf_log(1 - p)


log_2_pi_e = np.log(2.0 * np.pi * np.e)


@tf.function
def entropy_normal_from_logvar(logvar):
    return 0.5 * (log_2_pi_e + logvar)


def softmax_multi_with_log(x, single_values=3, temperature=10.0):
    """
    Compute softmax values for each sets of scores in x.
    """

    x = x.reshape(-1, single_values)
    x = x - np.max(x, 1).reshape(-1, 1)  # Normalization
    e_x = np.exp(x / temperature)
    SM = e_x / e_x.sum(axis=1).reshape(-1, 1)
    log_base = e_x.sum(axis=1).reshape(-1, 1)
    logSM = x - stable_np_log(log_base)
    return SM, logSM


def total_correlation(data):
    Cov = np.cov(data.T)
    return 0.5 * (np.log(np.diag(Cov)).sum() - np.linalg.slogdet(Cov)[1])


def compute_omega(loss_habitual, omega_params):
    """
    Computes precision factor Ω.
    In cognitive terms, it can be taught of as top-down attention. Effect is to incentivize disentanglement of latent state.
    a: The sum a+d show the maximum value of omega
    b: This shows the average value of D_kl[pi] that will cause half sigmoid (i.e. d+a/2)
    c: This moves the steepness of the sigmoid
    d: This is the minimum omega (when sigmoid is zero)
    """
    a, b, c, d = [omega_params[key] for key in omega_params]

    return a * (1.0 - 1.0 / (1.0 + np.exp(-(loss_habitual.numpy() - b) / c))) + d


def action_to_multi_hot(action_index, dtype):
    # Possible actions for slimevolley:
    #   0: nothing
    #   1: forward
    #   2: jump
    #   3: backward
    #   4: forward + jump
    #   5: backward + jump

    # Expected action format: multi-hot [forward, backward, jump]

    multi_hot_action = [0, 0, 0]

    # Flip forward if needed
    if action_index in [1, 4]:
        multi_hot_action[0] = 1

    # Flip backward if needed
    if action_index in [3, 5]:
        multi_hot_action[1] = 1

    # Flip jump if needed
    if action_index in [2, 4, 5]:
        multi_hot_action[2] = 1

    return tf.convert_to_tensor(multi_hot_action, dtype=dtype)
