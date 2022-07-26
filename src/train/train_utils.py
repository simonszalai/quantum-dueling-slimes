import numpy as np
from time import time
from datetime import timedelta

import src.train.config as cfg


class ProgressLogger:
    def __init__(self, epochs):
        self.epochs = epochs
        self.start_time = time()
        self.epoch_start_time = None
        self.epoch_times = []

    def start_epoch(self):
        self.epoch_start_time = time()

    def print(self, epoch, round_):
        now = time()
        epoch_time = round(now - self.epoch_start_time, 2)
        self.epoch_times.append(epoch_time)
        avg_epoch_time = np.array(self.epoch_times).mean()
        epochs_left = self.epochs - epoch
        approx_time_left = avg_epoch_time * epochs_left
        time_left = timedelta(seconds=round(approx_time_left))

        elapsed_seconds = now - self.start_time
        elapsed_time = timedelta(seconds=round(elapsed_seconds))

        print(f"Epoch {epoch} done in {'%.2f' % epoch_time} s ({round_} rounds) | Time left: ~ {str(time_left)} | Elapsed time: {str(elapsed_time)}")


def init_epoch(env, logger):
    # Reset environment and get initial observation for the agent
    obs_agent = env.reset()

    # Initial observation for agent and opponent is the same
    obs_opponent = obs_agent

    # Keras layers require a dimension for batches even if it equals to 1
    obs_agent = np.expand_dims(obs_agent, axis=0)

    # Initialize some variables
    round = 0
    done = False
    total_reward = 0

    # Start timer for epoch
    logger.start_epoch()

    return obs_agent.astype(cfg.np_precision), obs_opponent, round, done, total_reward


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
