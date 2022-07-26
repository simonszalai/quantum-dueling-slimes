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
