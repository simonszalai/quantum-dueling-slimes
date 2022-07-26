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

parser = argparse.ArgumentParser(description="Inference script.")
parser.add_argument("-nc", "--network-classical", type=str, default="", required=True, help="The path of a checkpoint to be loaded to the classical network.")
parser.add_argument("-nq", "--network-quantum", type=str, default="", required=False, help="The path of a checkpoint to be loaded to the quantum network.")
parser.add_argument("-ea", "--export-actions", type=str, default="", required=False, help="Name of the run where logs of executed actions should be exported.")


start_time = time.time()


def get_elapsed_time():
    now = time.time()
    elapsed_s = now - start_time
    return f"{round(elapsed_s * 1000, 1)} ms"


def tprint(text):
    print(text, get_elapsed_time())


args = parser.parse_args()

# Remove trailing slash from network path if it's there
if args.network_classical[-1] == "/":
    args.network_classical = args.network_classical[:-1]

if args.network_quantum and args.network_quantum[-1] == "/":
    args.network_quantum = args.network_quantum[:-1]

# Set up SlimeVolley Environment
policy = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
env = gym.make("SlimeVolley-v0")
env.seed(42)
# env.seed(np.random.randint(0, 10000))

model_classical = ActiveInferenceModel(model_type="classical")
model_classical.load(Path(args.network_classical))

if args.network_quantum:
    model_quantum = ActiveInferenceModel(model_type="quantum")
    model_quantum.load(Path(args.network_quantum))


env.render()
obs_agent = env.reset()
obs_opponent = obs_agent
obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

if args.network_quantum:
    obs_opponent = np.expand_dims(obs_opponent, axis=0)

steps = 0
total_reward = 0
action_opponent = np.array([0, 0, 0])

done = False
actions_register = []

while not done:
    action_agent_index, _ = model_classical.predict_agent_action(obs_agent, use_mcts=True)
    action_agent_multihot = utils.action_to_multi_hot(action_agent_index)

    if args.network_quantum:
        action_opponent_index, _ = model_quantum.predict_agent_action(obs_opponent, use_mcts=True)
        action_opponent_multihot = utils.action_to_multi_hot(action_opponent_index)
    else:
        action_opponent_multihot, _ = policy.predict(obs_opponent)

    obs_agent, reward, done, info = env.step(action_agent_multihot, action_opponent_multihot)

    actions_register.append({"action_agent_multihot": action_agent_multihot.numpy().tolist(), "action_opponent_multihot": action_opponent_multihot})

    obs_opponent = info["otherObs"]
    if args.network_quantum:
        obs_opponent = np.expand_dims(obs_opponent, axis=0)
    obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

    total_reward += reward

    env.render()
    sleep(0.04)  # 0.01

env.close()
print("cumulative score", total_reward)

if args.export_actions:
    with open(f"run_files/{args.export_actions}.json", "w") as f:
        json.dump(actions_register, f)
