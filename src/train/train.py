import os
import gym
import time
import argparse
import slimevolleygym
import numpy as np
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
    omega_params,
    calc_G_steps_ahead,
    average_G_over_N_samples,
)
from src.utils import compute_omega, softmax_multi_with_log, action_to_multi_hot
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


epoch_times = []
start_time = time.time()
print("=================================")
print(f"Training started: {epochs} epochs")
print("=================================")

for epoch in range(start_epoch, epochs + 1):
    # TODO: this doesnt work properly when loaded from checkpoint
    if epoch > gamma_delay and model.encoder_net.gamma < gamma_max:
        model.encoder_net.gamma.assign(model.encoder_net.gamma + gamma_rate)

    obs_0 = env.reset()
    obs_0 = np.expand_dims(obs_0, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

    total_reward = 0
    opponent_action = np.array([0.0, 0.0, 0.0])
    done = False

    i = 0
    epoch_start_time = time.time()

    while not done:
        print(f"Round {i} of epoch {epoch}\r", end="")
        i += 1

        # TRSTEP 3 Run planner and compute prior policy P (at)
        # TRSTEP 3.a Define shape of action space
        dummy_action_onehot = tf.eye(action_dim, dtype=np_precision)  # Shape: (action_counts, action_counts), e.g. (3, 3)

        # TRSTEP 3.b Compute Expected Free Energy
        # samples: average G over N samples
        o0_repeated = obs_0.repeat(action_dim, 0)

        sum_G = model.calculate_G_repeated(
            o0_repeated, dummy_action_onehot, steps=calc_G_steps_ahead, average_G_over_N_samples=average_G_over_N_samples, calc_mean=True
        )  # Shape (batch * action_counts,), e.g. (3,)
        # TRSTEP 3.c Compute prior policy (probability distribution over actions)
        P_action, log_P_action = softmax_multi_with_log(-sum_G.numpy(), action_dim)  # Shape: (batch, action_dim), e.g. (1, 3)

        # TRSTEP 9 Apply action a ̃ ∼ P (a ) to the environment.
        # TRSTEP 9.a Sample prior policy (action probability distributions)
        action_index = np.random.choice(action_dim, p=P_action.squeeze(axis=0))

        # Convert action index to one-hot (for network training)
        agent_action_onehot = np.zeros((1, action_dim), dtype=np_precision)
        agent_action_onehot[0, action_index] = 1.0

        # Convert action choices to multi-hot (for environment)
        agent_action = action_to_multi_hot(action_index, dtype=model.tf_precision)

        # TRSTEP 9.c Apply actions to the environment.
        opponent_action = policy.predict(obs_0.squeeze(axis=0))
        # Action format: multi-hot [forward, backward, jump]
        obs_1, reward, done, _ = env.step(agent_action, opponent_action)
        obs_1 = np.expand_dims(obs_1, axis=0)  # Keras layers requires a dimension for batches even if it equals to 1

        # -- TRAIN HABITUAL NETWORK ---------------------------------------------------
        # 4. Compute Qφs (st) using o ̃t.
        state_0, _, _ = model.encoder_net.encode_with_sample(obs_0)

        loss_habitual = model.habitual_net.train(state_0, P_action)
        TENSORBOARD.loss_habitual(loss_habitual)

        # -- TRAIN TRANSITION NETWORK ------------------------------------------------
        # 11. Compute Qφs (st+1 ) using o ̃t+1 .
        state_1_mean, state_1_logvar = model.encoder_net.encode(obs_1)
        loss_transition, pred_state_1_mean, pred_state_1_logvar = model.transition_net.train(
            state_0, agent_action_onehot, state_1_mean=state_1_mean, state_1_logvar=state_1_logvar, omega=model.omega
        )
        TENSORBOARD.loss_transition(loss_transition)

        current_omega = compute_omega(loss_habitual, omega_params=omega_params).reshape(-1, 1)
        model.omega.assign(tf.reduce_mean(current_omega))
        TENSORBOARD.omega(model.omega)

        # -- TRAIN ENCODER NETWORK --------------------------------------------------
        loss_encoder = model.encoder_net.train(obs_1=obs_1, pred_state_1_mean=pred_state_1_mean, pred_state_1_logvar=pred_state_1_logvar, omega=current_omega)
        TENSORBOARD.loss_encoder(loss_encoder)
        TENSORBOARD.gamma(model.encoder_net.gamma)

        TENSORBOARD.write_all_metrics(step=epoch * epochs + i)

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

    if i % 5 == 0:
        model.save(training_run_path / "saved_models" / f"epoch_{epoch}")

    print(f"Epoch {epoch} done in {'%.2f' % epoch_time} s ({i} rounds) | Time left: ~ {str(time_left)} | Elapsed time: {str(elapsed_time)}")
