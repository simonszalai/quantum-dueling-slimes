import numpy as np
import tensorflow as tf

import src.train.config as cfg


def calc_action_threshold(P, axis):
    """
    Calculates threshold to decide if the most likely action is distinct enough from the rest
    """
    return np.max(P, axis=axis) - np.mean(P, axis=axis)


def normalize_distribution(x):
    return x / tf.reduce_sum(x, axis=0)


@tf.function
def select_action_from_dist(action_probs, deterministic):
    if deterministic:
        # Choose the action with the highest probability
        # action_index = np.argmax(action_probs)
        action_index = tf.math.argmax(action_probs).numpy()
    else:
        # Sample from the actions probability distribution
        # action_index = np.random.choice(cfg.action_dim, p=action_probs)
        action_index = tf.random.categorical(tf.math.log(action_probs), 1)

    return action_index
