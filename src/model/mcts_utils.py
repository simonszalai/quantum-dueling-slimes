import numpy as np

import src.train.config as cfg


def calc_action_threshold(P, axis):
    """
    Calculates threshold to decide if the most likely action is distinct enough from the rest
    """
    return np.max(P, axis=axis) - np.mean(P, axis=axis)


def normalize_distribution(x):
    return x / x.sum(axis=0)


def select_action_from_dist(action_probs, deterministic):
    if deterministic:
        # Choose the action with the highest probability
        action_index = np.argmax(action_probs)
    else:
        # Sample from the actions probability distribution
        action_index = np.random.choice(cfg.action_dim, p=action_probs)

    return action_index
