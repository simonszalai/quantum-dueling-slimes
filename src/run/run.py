import os
import gym
import time
import argparse
import slimevolleygym
from time import sleep
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.run.mcts import MCTS
from src.model.active_inference import ActiveInferenceModel


tf.config.run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument("-n", "--network", type=str, default="", required=True, help="The path of a checkpoint to be loaded.")

start_time = time.time()


def get_elapsed_time():
    now = time.time()
    elapsed_s = now - start_time
    return f"{round(elapsed_s * 1000, 1)} ms"


def tprint(text):
    print(text, get_elapsed_time())


# Whether expected free energy should be calculated using the mean instead of sampling
mean_instead_sampling = False

# Define 3 variables that determine the number of iterations on each level (run, experiment, round)
total_timesteps_count = 50000
timesteps_in_experiment = 1000
timesteps_in_round = 100

# Select method used by the agent for action selection. Available methods: mcts (rest disabled for simplicity)
action_selection_method = "mcts"

# Parameter to define the 'softness' of the softmax function. e.g.:
# - Sample 'hard' softmax probs : [0.01,0.01,0.98]
# - Sample 'soft' softmax probs : [0.2,0.2,0.6]
# 1 means no change to normal softmax output, closer to 0 it gets harder
softmax_temperature = 1

# For each action, move this many pixels to the selected direction
apply_same_action_N_times = 15

args = parser.parse_args()

# Remove trailing slash from network path if it's there
if args.network[-1] == "/":
    args.network = args.network[:-1]


mcts = MCTS()

# Set up SlimeVolley Environment
policy = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
env = gym.make("SlimeVolley-v0")
env.seed(np.random.randint(0, 10000))

tprint("Environment loaded")

model = ActiveInferenceModel()
tprint("Active Inference model instantiated")
model.load(Path(args.network))
tprint("Active Inference model checkpoint loaded")

env.render()
obs_agent = env.reset()
obs_opponent = obs_agent
obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

steps = 0
total_reward = 0
action_opponent = np.array([0, 0, 0])

done = False


while not done:
    action_agent, _ = model.predict_agent_action(obs_agent)
    action_opponent = policy.predict(obs_opponent)

    obs_agent, reward, done, info = env.step(action_agent, action_opponent)

    obs_opponent = info["otherObs"]
    obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

    total_reward += reward

    env.render()
    sleep(0.01)  # 0.01

env.close()
print("cumulative score", total_reward)
