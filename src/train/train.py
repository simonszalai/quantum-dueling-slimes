import os
import gym
import time
import argparse
import slimevolleygym
import numpy as np
from time import sleep
import tensorflow as tf
from pathlib import Path
from datetime import datetime, timedelta

from src.train.config import (
    state_dim,
    action_dim,
    np_precision,
    beta_state,
    beta_obs,
    gamma,
    gamma_rate,
    gamma_max,
    gamma_delay,
    learning_rates,
)
from src.train.metrics import TENSORBOARD
from src.model.active_inference import ActiveInferenceModel

# Set TensorFlow logging level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Parse CLI arguments
parser = argparse.ArgumentParser(description="Training script.")
parser.add_argument(
    "-r",
    "--resume",
    type=str,
    help="If this is used, the script tries to load existing weights and resume training from the path specified (within the checkpoints directory)",
)
parser.add_argument("-p", "--path", type=str, default="", help="Path to save training logs, checkpoints and the trained model.")
parser.add_argument("-e", "--epochs", type=int, default=1000, help="Length of training.")
parser.add_argument("-b", "--batch", type=int, default=1, help="Select batch size.")
args = parser.parse_args()

# Set folder names for saving training logs, then create them if they don't exist
# training_id = args.resume.split("/")[0] if args.resume else datetime.now().strftime("%Y%m%d-%H%M%S")
training_base_path = Path(args.path) if args.path else Path("training_files")
training_id = args.resume if args.resume else datetime.now().strftime("%Y%m%d-%H%M%S")
training_run_path = training_base_path / training_id
os.makedirs(training_run_path, exist_ok=True)

# ========= DEBUG SECTION =========
# Enable eager execution in @tf.functions
# Slows down training a lot (~10-20x), but allows retrieving (and printing) actual tensor value
# ALTERNATIVE: use tf.print instead, which doesn't require eager execution to print values
tf.config.run_functions_eagerly(True)
np.set_printoptions(threshold=1000)

# Enable TensorFlow Debugger V2.
# Quite big debugging overhead, but can help catching hard-to-find bugs (also gives an error when saving the file, so use it only for debugging!)
# tf.debugging.experimental.enable_dump_debug_info(Path(training_base_path, "logs").as_posix(), tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# ========= /DEBUG SECTION =========

# Create TensorBoard log writer
# To access logs, start TensorBoard: tensorboard --logdir dueling_active_inference/dsprites/training_files/logs --samples_per_plugin scalars=999999999
training_id = args.resume.split("/")[0] if args.resume else datetime.now().strftime("%Y%m%d-%H%M%S")
TENSORBOARD.create_writer(training_run_path)

# Define training length
epochs = args.epochs

model = ActiveInferenceModel(
    state_dim=state_dim,
    action_dim=action_dim,
    gamma=gamma,
    beta_state=beta_state,
    beta_obs=beta_obs,
    learning_rates=learning_rates,
    training_run_path=training_run_path,
    np_precision=np_precision,
)

# Resume training
start_epoch = 0
if args.resume and model.checkpoint_manager.latest_checkpoint:
    start_epoch = int(model.checkpoint_manager.latest_checkpoint.split("-")[-1])
    restored_path = model.resume_training(training_run_path / "checkpoints")
    print(f"Restored to epoch {start_epoch} from {restored_path}")


# Set up SlimeVolley Environment
policy = slimevolleygym.BaselinePolicy()  # defaults to use RNN Baseline for player
env = gym.make("SlimeVolley-v0")
env.seed(np.random.randint(0, 10000))

render_mode = False

epoch_times = []
start_time = time.time()
print("=================================")
print(f"Training started: {epochs} epochs")
print("=================================")

for epoch in range(start_epoch, epochs + 1):
    # TODO: this doesnt work properly when loaded from checkpoint
    if epoch > gamma_delay and model.encoder_net.gamma < gamma_max:
        model.encoder_net.gamma.assign(model.encoder_net.gamma + gamma_rate)

    if render_mode:
        env.render()

    obs_agent = env.reset()
    obs_opponent = obs_agent
    obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

    total_reward = 0
    action_opponent = np.array([0.0, 0.0, 0.0])
    done = False

    i = 0
    epoch_start_time = time.time()

    while not done:
        print(f"Round {i} of epoch {epoch}\r", end="")
        i += 1

        action_agent, train_info = model.predict_agent_action(obs_agent)
        action_index, P_action, agent_action_onehot = train_info

        # Get action of the opponent
        action_opponent = policy.predict(obs_opponent)

        model.train(obs_agent, train_info, step=epoch * epochs + i)

        # TRSTEP 9.c Apply actions to the environment.
        # Action format: multi-hot [forward, backward, jump]
        obs_agent, reward, done, info = env.step(action_agent, action_opponent)

        obs_opponent = info["otherObs"]
        obs_agent = np.expand_dims(obs_agent, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

        if render_mode:
            env.render()
            sleep(0.01)

    # model.create_checkpoint()
    now = time.time()
    epoch_time = round(now - epoch_start_time, 2)
    epoch_times.append(epoch_time)
    avg_epoch_time = np.array(epoch_times).mean()
    epochs_left = epochs - epoch
    approx_time_left = avg_epoch_time * epochs_left
    time_left = timedelta(seconds=round(approx_time_left))

    elapsed_seconds = now - start_time
    elapsed_time = timedelta(seconds=round(elapsed_seconds))

    if epoch != 0 and epoch % 25 == 0:
        model.save(training_run_path / "saved_models" / f"epoch_{epoch}")

    print(f"Epoch {epoch} done in {'%.2f' % epoch_time} s ({i} rounds) | Time left: ~ {str(time_left)} | Elapsed time: {str(elapsed_time)}")
