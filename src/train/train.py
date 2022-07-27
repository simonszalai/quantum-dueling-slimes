import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Only log tf  errors (has to be before tf import)
import gym
import argparse
import slimevolleygym
import numpy as np
from time import sleep
import tensorflow as tf
from pathlib import Path
from datetime import datetime

import src.train.train_utils as train_utils
import src.train.config as cfg
from src.train.config import gamma_rate, gamma_max, gamma_delay
from src.train.metrics import TENSORBOARD
from src.model.active_inference import ActiveInferenceModel


# ========= DEBUG SECTION =========
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# Enable eager execution in @tf.functions
# Slows down training a lot (~10-20x), but allows retrieving (and printing) actual tensor value
# ALTERNATIVE: use tf.print instead, which doesn't require eager execution to print values
tf.config.run_functions_eagerly(False)
np.set_printoptions(threshold=1000)

# Enable TensorFlow Debugger V2.
# Quite big debugging overhead, but can help catching hard-to-find bugs (also throws an error when saving the file, so use it only for debugging!)
# tf.debugging.experimental.enable_dump_debug_info(Path(training_base_path, "logs").as_posix(), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# ========= /DEBUG SECTION =========


# Parse CLI arguments
parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument("-r", "--render", action="store_true", help="Enables showing the gym environment during training.")
parser.add_argument("-p", "--path", type=str, default="", help="Path to save training logs, checkpoints and the trained model.")
parser.add_argument("-e", "--epochs", type=int, default=300, help="Length of training.")
args = parser.parse_args()

# Set folder names for saving training logs, then create them if they don't exist
training_base_path = Path("training_files")
training_id = Path(args.path) if args.path else datetime.now().strftime("%Y%m%d-%H%M%S")
training_run_path = training_base_path / training_id
os.makedirs(training_run_path, exist_ok=True)

# Create TensorBoard log writer (see README.md how to view logs)
training_id = datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD.create_writer(training_run_path)

# Set up SlimeVolley Environment
policy = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
policy_trainer = slimevolleygym.BaselinePolicy()
env = gym.make("SlimeVolley-v0")
env.seed(42)

# Set up active inference instance
logger = train_utils.ProgressLogger(args.epochs)
model = ActiveInferenceModel(training_run_path=training_run_path, add_quantum_noise=False)


print("=================================")
print(f"Training started: {args.epochs} epochs")
print("=================================")

for epoch in range(0, args.epochs + 1):
    if args.render:
        env.render()

    # Update gamma
    if epoch > gamma_delay and model.encoder_net.gamma < gamma_max:
        model.encoder_net.gamma.assign(model.encoder_net.gamma + gamma_rate)

    # Start new epoch (new game in SlimeVolley)
    obs_0_agent, obs_0_opponent, round_, done, total_reward = train_utils.init_epoch(env, logger)

    while not done:
        print(f"Round {round_} of epoch {epoch}\r", end="")
        round_ += 1

        # Get action of the agent (active inference)
        action_agent_multihot, action_agent_onehot = model.predict_agent_action(obs_0_agent)

        # Get action of the opponent (baseline RNN policy)
        action_opponent_multihot, _ = policy.predict(obs_0_opponent)

        # Get action of another baseline policy, that the habitual network can learn to mimic
        _, P_action_trainer = policy_trainer.predict(obs_0_agent.astype(cfg.np_precision).squeeze())
        P_action_trainer = train_utils.convert_env_P_action_to_active_inference_format(P_action_trainer)

        # Apply actions to the environment. Action format: multi-hot [forward, backward, jump]
        obs_1_agent, reward, done, info = env.step(action_agent_multihot, action_opponent_multihot)
        obs_1_agent = np.expand_dims(obs_1_agent.astype(cfg.np_precision), axis=0)

        # Train model
        step = epoch * args.epochs + round_
        model.train(obs_0_agent, obs_1_agent, action_agent_onehot, P_action_trainer, step=step)

        # Update observations for next round
        obs_0_agent = obs_1_agent
        obs_0_opponent = info["otherObs"]

        if args.render:
            env.render()
            sleep(0.01)

    if epoch != 0 and epoch % 25 == 0:
        model.save(training_run_path / "saved_models" / f"epoch_{epoch}")

    logger.print(epoch, round_)
