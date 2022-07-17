import os
import gym
import argparse
import slimevolleygym
import numpy as np
from time import sleep
import tensorflow as tf
from pathlib import Path
from datetime import datetime

from src.train.config import gamma_rate, gamma_max, gamma_delay
from src.train.metrics import TENSORBOARD
from src.model.active_inference import ActiveInferenceModel
from src.train.train_utils import ProgressLogger, init_epoch


# ========= DEBUG SECTION =========
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# Enable eager execution in @tf.functions
# Slows down training a lot (~10-20x), but allows retrieving (and printing) actual tensor value
# ALTERNATIVE: use tf.print instead, which doesn't require eager execution to print values
tf.config.run_functions_eagerly(True)
np.set_printoptions(threshold=1000)

# Enable TensorFlow Debugger V2.
# Quite big debugging overhead, but can help catching hard-to-find bugs (also throws an error when saving the file, so use it only for debugging!)
# tf.debugging.experimental.enable_dump_debug_info(Path(training_base_path, "logs").as_posix(), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# ========= /DEBUG SECTION =========


# Parse CLI arguments
parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument("-r", "--render", action="store_true", help="Enables showing the gym environment during training.")
parser.add_argument("-p", "--path", type=str, default="", help="Path to save training logs, checkpoints and the trained model.")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="Length of training.")
parser.add_argument("-b", "--batch", type=int, default=1, help="Select batch size.")
args = parser.parse_args()

# Set folder names for saving training logs, then create them if they don't exist
training_base_path = Path(args.path) if args.path else Path("training_files")
training_id = datetime.now().strftime("%Y%m%d-%H%M%S")
training_run_path = training_base_path / training_id
os.makedirs(training_run_path, exist_ok=True)

# Create TensorBoard log writer (see README.md how to view logs)
training_id = datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD.create_writer(training_run_path)

# Set up SlimeVolley Environment
policy = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
env = gym.make("SlimeVolley-v0")
env.seed(np.random.randint(0, 10000))

# Set up active inference instance
logger = ProgressLogger(args.epochs)
model = ActiveInferenceModel(training_run_path=training_run_path)


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
    obs_agent, obs_opponent, round_, done, total_reward = init_epoch(env, logger)

    while not done:
        print(f"Round {round_} of epoch {epoch}\r", end="")
        round_ += 1

        # Get action of the agent
        action_agent, train_info = model.predict_agent_action(obs_agent)
        action_index, P_action, agent_action_onehot = train_info

        # Get action of the opponent
        action_opponent = policy.predict(obs_opponent)

        # Train model
        model.train(obs_agent, train_info, step=epoch * args.epochs + round_)

        # Apply actions to the environment. Action format: multi-hot [forward, backward, jump]
        obs_agent, reward, done, info = env.step(action_agent, action_opponent)

        # Update/format observations for next round
        obs_opponent = info["otherObs"]
        obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

        if args.render:
            env.render()
            sleep(0.01)

    if epoch != 0 and epoch % 25 == 0:
        model.save(training_run_path / "saved_models" / f"epoch_{epoch}")

    logger.print(epoch, round_)
