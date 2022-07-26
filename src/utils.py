import numpy as np
import tensorflow as tf
import src.train.config as cfg

eps = np.finfo(cfg.np_precision).eps * 10


def stable_tf_log(x):
    return tf.math.log(x + eps)


def stable_np_log(x):
    return np.log(x + eps)


@tf.function
def reparameterize(mean, logvar):
    """
    Essential to make backpropagation possible despite the fact that random sampling occurs
    """
    eps = tf.random.normal(shape=mean.shape)
    std = tf.exp(logvar * 0.5)
    return eps * std + mean


def D_KL_from_logvar_and_precision(state_1_mean, state_1_logvar, state_2_mean, state_2_logvar, omega):
    """
    Computes the difference between two distributions
    """
    D_KL = (
        0.5 * (state_2_logvar - stable_tf_log(omega) - state_1_logvar)
        + (tf.exp(state_1_logvar) + tf.math.square(state_1_mean - state_2_mean)) / (2.0 * tf.exp(state_2_logvar) / omega)
        - 0.5
    )
    return tf.reduce_sum(D_KL, 1)


def entropy_bernoulli(p):
    return -(1 - p) * stable_tf_log(1 - p) - p * stable_tf_log(p)


@tf.function
def entropy_gaussian(logvar):
    log_2_pi_e = np.log(2.0 * np.pi * np.e)
    return 0.5 * (log_2_pi_e + logvar)


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


def action_to_multi_hot(action_index):
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

    return tf.convert_to_tensor(multi_hot_action, dtype=cfg.np_precision)


def action_to_onehot(action_index):
    # Convert action index to one-hot encoding (for network training)
    action_onehot = np.zeros((1, cfg.action_dim), dtype=cfg.np_precision)
    action_onehot[0, action_index] = 1.0

    return action_onehot


def norm_tanh_range(n):
    """
    Changes from range of tanh [-1, 1] to range of [0, 1]
    """
    return (n + 1) / 2


def get_idle_chance(P_action):
    """
    First it Inverts the probability distribution (chance that each action does not happen),
    than calculates the product of them (chance that none of those are happening)
    """
    inv_P_action = 1 - P_action
    return np.prod(inv_P_action)


def convert_env_P_action_to_active_inference_format(env_P_action):
    """
    Parameters
    ----------
    env_P_action : np.array
        action probability distributions of the baseline policy, extracted from slimevolley package. Since a tanh activation is applied, range is [-1, 1].

    Converts probabilities from multihot-like format
        [forward, backward, jump]
    to onehot-like format
        [idle, forward, jump, backward, forward + jump, backward + jump]
    """
    norm_env_P_action = norm_tanh_range(env_P_action.squeeze())

    P_action = np.zeros(
        (
            1,
            cfg.action_dim,
        )
    )
    P_action[0][0] = get_idle_chance(norm_env_P_action)  # No action taken
    P_action[0][1] = norm_env_P_action[0]  # Forward
    P_action[0][2] = norm_env_P_action[2]  # Jump
    P_action[0][3] = norm_env_P_action[1]  # Backward
    P_action[0][4] = norm_env_P_action[0] + norm_env_P_action[2]  # Forward + Jump
    P_action[0][5] = norm_env_P_action[1] + norm_env_P_action[2]  # Backward + Jump

    # Re-normalize
    P_action /= P_action.sum()

    return P_action.astype(cfg.np_precision)


def multihot_select_action(multihot_state):
    forward = 0
    backward = 0
    jump = 0
    if multihot_state[0] > 0.75:
        forward = 1
    if multihot_state[1] > 0.75:
        backward = 1
    if multihot_state[2] > 0.75:
        jump = 1
    return [forward, backward, jump]
