import os
import gym
import json
import argparse
import slimevolleygym
from time import sleep
import numpy as np
import tensorflow as tf
from pathlib import Path

from src.model.active_inference import ActiveInferenceModel


np.set_printoptions(threshold=1000)
tf.config.run_functions_eagerly(True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

parser = argparse.ArgumentParser(description="Inference script.")
parser.add_argument("-n", "--network", type=str, default="", required=True, help="The path of a checkpoint to be loaded to the trained network.")
parser.add_argument("-ea", "--export-actions", type=str, default="", required=False, help="Name of the run where logs of executed actions should be exported.")
args = parser.parse_args()

# Remove trailing slash from network path if it's there
if args.network[-1] == "/":
    args.network = args.network[:-1]

# Set up SlimeVolley Environment
policy = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
env = gym.make("SlimeVolley-v0")
env.seed(42)

# Set up two active inference agents
model_agent = ActiveInferenceModel(add_quantum_noise=False, use_habit=True, using_prior_for_exploration=True)
model_agent.load(Path(args.network))

model_opponent = ActiveInferenceModel(add_quantum_noise=True, use_habit=True, using_prior_for_exploration=True)
model_opponent.load(Path(args.network))

# Initialize environment and get first observations
env.render()
obs_agent = env.reset()
obs_opponent = obs_agent
obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1
obs_opponent = np.expand_dims(obs_opponent, axis=0)

# Init registers
done = False
total_reward = 0
actions_register = []

while not done:
    # Get actions from agent and opponent
    action_agent_multihot, _ = model_agent.predict_agent_action(obs_agent, use_mcts=True)
    action_opponent_multihot, _ = model_opponent.predict_agent_action(obs_opponent, use_mcts=True)

    # Apply actions to the environemnt
    obs_agent, reward, done, info = env.step(action_agent_multihot, action_opponent_multihot)

    # Add actions to register so they can be saved and replayed separately
    actions_register.append(
        {"action_agent_multihot": action_agent_multihot.numpy().tolist(), "action_opponent_multihot": action_opponent_multihot.numpy().tolist()}
    )

    # Assign observations in right format for the next round
    obs_opponent = np.expand_dims(info["otherObs"], axis=0)
    obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

    total_reward += reward

    env.render()
    sleep(0.04)  # 0.01

env.close()
print("cumulative score", total_reward)

if args.export_actions:
    with open(f"run_files/{args.export_actions}.json", "w") as f:
        json.dump(actions_register, f)
