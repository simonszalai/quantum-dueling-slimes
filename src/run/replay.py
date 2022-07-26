import os
import gym
import json
import time
import argparse
import slimevolleygym
from time import sleep
import numpy as np
import tensorflow as tf
from pathlib import Path

import src.utils as utils
from src.model.active_inference import ActiveInferenceModel


np.set_printoptions(threshold=1000)
tf.config.run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument(
    "-ia", "--import-actions", type=str, default="", required=False, help="Name of the run where logs of executed actions should be imported from."
)
args = parser.parse_args()

env = gym.make("SlimeVolley-v0")
env.seed(42)

env.render()
obs_agent = env.reset()
obs_opponent = obs_agent
obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

with open(f"run_files/{args.import_actions}.json") as f:
    actions = json.load(f)

step = 0
total_reward = 0
done = False

while not done:
    actions_of_step = actions[step]

    obs_agent, reward, done, info = env.step(actions_of_step["action_agent_multihot"], actions_of_step["action_opponent_multihot"])

    total_reward += reward

    env.render()
    sleep(0.05)  # 0.01
    step += 1

env.close()
print("cumulative score", total_reward)
