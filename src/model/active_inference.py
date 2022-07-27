import numpy as np
import tensorflow as tf

import src.utils as utils
import src.train.config as cfg
from src.model.habitual_network_quantum import HabitualNetworkQuantum
from src.model.transition_network import TransitionNetwork
from src.model.encoder_network import EncoderNetwork
from src.model.mcts import MCTS
from src.train.metrics import TENSORBOARD


class ActiveInferenceModel:
    def __init__(self, add_quantum_noise=False, training_run_path=None, use_habit=False, using_prior_for_exploration=True):
        self.omega = tf.Variable(1.0, trainable=False, name="omega")

        tf.keras.backend.set_floatx(f"float{np.finfo(cfg.np_precision).bits}")

        self.habitual_net = HabitualNetworkQuantum(add_quantum_noise=add_quantum_noise)
        self.transition_net = TransitionNetwork()
        self.encoder_net = EncoderNetwork()
        self.mcts = MCTS(model=self, use_habit=use_habit, using_prior_for_exploration=using_prior_for_exploration)

        self.checkpoint = tf.train.Checkpoint(
            habitual_net=self.habitual_net,
            transition_net=self.transition_net,
            encoder_net=self.encoder_net,
            omega=self.omega,
            beta_state=self.encoder_net.beta_state,
            beta_obs=self.encoder_net.beta_obs,
            gamma=self.encoder_net.gamma,
        )

        # Only needed for training
        if training_run_path:
            self.checkpoints_path = training_run_path / "checkpoints"
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.checkpoints_path.as_posix(), max_to_keep=5)

    def check_reward(self, obs_for_actions):
        """
        Calculate reward for each action

        obs_for_actions : observation for each possible action
        """
        # obs:
        # Smaller reward is preferred
        # negative X: opponents half
        # x_agent, y_agent, Vx_agent, Vy_agent, x_ball, y_ball, Vx_ball, Vy_ball, X_opponent, Y_opponent, Vx_opponent, Vy_opponent
        # [ 0.2    0.237    0.        1.252     -2.231  1.04    1.191    0.341    2.192       0.46         0.          -0.316]
        # [ 0.2    0.275    0.        1.154     -2.191  1.048   1.191    0.243    2.133       0.446       -1.75        -0.414]
        # [ 0.2    0.31     0.        1.056     -2.151  1.053   1.191    0.145    2.075       0.429       -1.75        -0.512]
        # [ 0.2    0.342    0.        0.958     -2.111  1.055   1.191    0.047    2.075       0.409        0.          -0.61 ]
        # [ 0.2    0.371    0.        0.86      -2.072  1.053   1.191   -0.051    2.017       0.385       -1.75        -0.708]
        # [ 0.2    0.396    0.        0.762     -2.032  1.048   1.191   -0.149    1.958       0.359       -1.75        -0.806]
        # [ 0.2    0.419    0.        0.664     -1.992  1.04    1.191   -0.247    1.958       0.328        0.          -0.904]
        # [ 0.2    0.437    0.        0.566     -1.953  1.028   1.191   -0.345    1.9         0.295       -1.75        -1.002]
        # [ 0.2    0.453    0.        0.468     -1.913  1.014   1.191   -0.443    1.842       0.258       -1.75        -1.1  ]
        # [ 0.2    0.465    0.        0.37      -1.873  0.996   1.191   -0.541    1.842       0.218        0.          -1.198]
        # [ 0.2    0.474    0.        0.272     -1.834  0.974   1.191   -0.639    1.783       0.175       -1.75        -1.296]
        # [ 0.2    0.48     0.        0.174     -1.794  0.95    1.191   -0.737    1.783       0.15         0.           0.   ]
        # [ 0.2    0.483    0.        0.076     -1.754  0.922   1.191   -0.835    1.725       0.195       -1.75         1.35 ]
        # [ 0.2    0.482    0.       -0.022     -1.714  0.891   1.191   -0.933    1.725       0.237        0.           1.252]
        # [ 0.2    0.478    0.       -0.12      -1.675  0.856   1.191   -1.031    1.725       0.275        0.           1.154]

        def get_agent_ball_dist(x_ball, y_ball, x_agent, y_agent):
            x_dist = tf.math.abs(x_agent - x_ball)
            y_dist = tf.math.abs(y_agent - y_ball)
            dist = tf.math.sqrt(tf.square(x_dist) + tf.square(y_dist))
            return dist

        actions_count = len(obs_for_actions)
        results = tf.TensorArray(cfg.tf_precision, size=actions_count)
        for i in tf.range(actions_count):
            obs = obs_for_actions[i]
            # x_agent = tf.gather(obs, 0)
            # y_agent = tf.gather(obs, 1)
            x_ball = tf.gather(obs, 4)
            y_ball = tf.gather(obs, 5)
            # Vx_ball = tf.gather(obs, 6)

            # We encode reward as expected outcome which is inversely proportional to the probability of the target observation given the ideal input policy
            # Modulates reward on the Y axis (the smaller it is, the reward is more concentrated on the bottom) range: [0,1]
            a = 0.5
            # Modulates reward on the X axis (the bigger it is, the more concentrated the reward is in the left and right corners) range: [0,1]
            b = 0.4
            free_energy = 10 * tf.math.exp(-y_ball / a) * tf.math.tanh(x_ball / b)

            results = results.write(i, free_energy)

        stacked_results = results.stack()
        return stacked_results

    @tf.function
    def habitual_network(self, obs):
        _, pred_state_mean, _ = self.encoder_net.encode(obs)
        P_action = self.habitual_net.predict_action(pred_state_mean)
        return P_action

    @tf.function
    def calculate_G(self, states_0, actions, average_G_over_N_samples=1):
        """
        Calculates Expected Free Energy of a given action taken in a given state.
        Takes multiple states and actions and calculates G in a batch.
        """

        assert (
            states_0.shape[0] == actions.shape[0]
        ), f"Sample count in states batch ({states_0.shape[0]}) must equal sample count in actions batch ({actions.shape[0]}) when calculating G."

        samples_in_batch = states_0.shape[0]

        # Create tensors in the right shape to accumulate the different terms of G (start with all zeros)
        term0 = tf.zeros([samples_in_batch], cfg.np_precision)
        term1 = tf.zeros([samples_in_batch], cfg.np_precision)
        term2_1 = tf.zeros(samples_in_batch, cfg.np_precision)
        term2_2 = tf.zeros(samples_in_batch, cfg.np_precision)

        # Calculate G 'average_G_over_N_samples' times
        for _ in range(average_G_over_N_samples):
            pred_states_1, pred_states_1_mean, pred_states_1_logvar = self.transition_net.transition_with_sample(states_0, actions)

            pred_obs_1 = self.encoder_net.decode(pred_states_1)
            _, _, encoded_pred_state_1_logvar = self.encoder_net.encode(pred_obs_1)

            # E [ log P(o|pi) ]
            log_pred_obs_1 = self.check_reward(pred_obs_1)
            term0 += log_pred_obs_1

            # E [ log Q(s|pi) - log Q(s|o,pi) ]
            term1_new = -tf.reduce_sum(
                utils.entropy_normal_from_logvar(pred_states_1_logvar) + utils.entropy_normal_from_logvar(encoded_pred_state_1_logvar), axis=1
            )
            term1 += term1_new

            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout
            pred_state_1_temp1, _, _ = self.transition_net.transition_with_sample(states_0, actions)
            pred_obs_1_temp1 = self.encoder_net.decode(pred_state_1_temp1)
            term2_1_new = tf.reduce_sum(utils.entropy_gaussian(pred_obs_1_temp1), axis=[1])
            term2_1 += term2_1_new

            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick
            pred_obs_temp2 = self.encoder_net.decode(pred_states_1)
            term2_2_new = tf.reduce_sum(utils.entropy_gaussian(pred_obs_temp2), axis=[1])
            term2_2 += term2_2_new

        # Calculate average for each term separately
        term0 /= float(average_G_over_N_samples)
        term1 /= float(average_G_over_N_samples)
        term2_1 /= float(average_G_over_N_samples)
        term2_2 /= float(average_G_over_N_samples)

        # Use the averaged terms to calculate batch of G's
        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2
        G = -term0 + term1 + term2

        # pred_state_1 is used in MCTS, it is assigned to child nodes as their state
        return G, pred_states_1, pred_states_1_mean

    @tf.function
    def calculate_G_repeated(self, obs, actions, steps=1, calc_mean=False, average_G_over_N_samples=1):
        """
        First encodes the starting observation to the starting state, calculates G for it given the passed action(s),
        then predicts the next state, calculates G from that state using the same action(s), repeating this 'steps' times

        Parameters
        ----------
        obs : np.array
            Observation of the agent before the passed actions are taken (will be encoded to state)
        actions : np.array
            Array of actions to compute G for
        steps : 1
            Number of times each actions should be repeated
        calc_mean : bool
            Whether expected free energy should be calculated using the mean instead of sampling
        average_G_over_N_samples : int
            Number of times the computation should be repeated then averaged over
        """

        # Predict current state given the passed observation
        state, state_mean, _ = self.encoder_net.encode(obs)

        # Init a register for G
        sum_G = tf.zeros([obs.shape[0]], cfg.np_precision)

        # Predict state_t+1
        state_temp = state_mean if calc_mean else state

        for t in range(steps):
            G, next_state, next_state_mean = self.calculate_G(state_temp, actions, average_G_over_N_samples=average_G_over_N_samples)
            sum_G += G
            state_temp = next_state_mean if calc_mean else next_state

        return sum_G

    @tf.function
    def calculate_G_given_trajectory(self, state_traj, pred_state_traj, pred_state_mean_traj, pred_state_logvar_traj, action_traj):
        # NOTE: len(s0_traj) = len(s1_traj) = len(pi0_traj)

        pred_obs = self.encoder_net.decode(pred_state_traj)
        _, _, pred_state_logvar = self.encoder_net.encode(pred_obs)

        # E [ log P(o|pi) ]
        term0 = self.check_reward(pred_obs)

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = -tf.reduce_sum(utils.entropy_normal_from_logvar(pred_state_logvar_traj) + utils.entropy_normal_from_logvar(pred_state_logvar), axis=1)

        #  Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout
        pred_obs_temp1 = self.encoder_net.decode(self.transition_net.transition_with_sample(action_traj, state_traj)[0])
        term2_1 = tf.reduce_sum(utils.entropy_gaussian(pred_obs_temp1), axis=[1])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparameterization trick
        pred_obs_temp2 = self.encoder_net.decode(utils.reparameterize(pred_state_mean_traj, pred_state_logvar_traj))
        term2_2 = tf.reduce_sum(utils.entropy_gaussian(pred_obs_temp2), axis=[1])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        return -term0 + term1 + term2

    def predict_agent_action(self, obs, use_mcts=False):
        if use_mcts:
            action_index = self.mcts.active_inference_mcts(obs)
            action_multihot = utils.action_to_multi_hot(action_index)
            action_agent_onehot = utils.action_to_onehot(action_index)
            return action_multihot, action_agent_onehot
        else:
            action_index, P_action = self.predict_agent_action_train(obs, deterministic=False)
            action_multihot = utils.action_to_multi_hot(action_index)
            action_agent_onehot = utils.action_to_onehot(action_index)
            return action_multihot, action_agent_onehot

    def predict_agent_action_train(self, obs, deterministic=False):
        # Repeat observation for each possible action
        # This will result in one predicted observation for each possible action so the lowest G can be calculated then the right action selected
        obs_repeated = obs.repeat(cfg.action_dim, axis=0)
        all_actions = tf.eye(cfg.action_dim, dtype=cfg.np_precision)  # Shape: (action_counts, action_counts), e.g. (3, 3)

        # Calculate a G value from the current observation for each possible action
        sum_G = self.calculate_G_repeated(obs_repeated, all_actions, steps=cfg.calc_G_steps_ahead, average_G_over_N_samples=cfg.average_G_over_N_samples)

        # Compute probability distribution of actions from G (smaller G -> bigger chance for action to be selected)
        P_action, _ = utils.softmax_multi_with_log(-sum_G.numpy(), cfg.action_dim)  # Shape: (batch, action_dim), e.g. (1, 3)

        # Sample an action from the probabilty distribution of actions
        # Deterministic flag added, not in original code
        if deterministic:
            action_index = np.argmax(P_action.squeeze(axis=0))
        else:
            action_index = np.random.choice(cfg.action_dim, p=P_action.squeeze(axis=0))

        return action_index, P_action

    def train(self, obs_0, obs_1, action_onehot, P_action, step):
        """
        Parameters
        ----------
        P_action : np.array
            Probability distribution over actions as predicted by the agent's internal model
        """

        # -- TRAIN HABITUAL NETWORK ---------------------------------------------------
        state_0, _, _ = self.encoder_net.encode(obs_0)

        loss_habitual = self.habitual_net.train(state_0, P_action)
        TENSORBOARD.loss_habitual(loss_habitual)

        # Update omega value
        current_omega = utils.compute_omega(loss_habitual, omega_params=cfg.omega_params).reshape(-1, 1)
        self.omega.assign(tf.reduce_mean(current_omega))
        TENSORBOARD.omega(self.omega)

        # -- TRAIN TRANSITION NETWORK ------------------------------------------------
        _, state_1_mean, state_1_logvar = self.encoder_net.encode(obs_1)
        loss_transition, pred_state_1_mean, pred_state_1_logvar = self.transition_net.train(
            state_0, action_onehot, actual_state_1_mean=state_1_mean, actual_state_1_logvar=state_1_logvar, omega=self.omega
        )
        TENSORBOARD.loss_transition(loss_transition)

        # -- TRAIN ENCODER NETWORK --------------------------------------------------
        loss_encoder = self.encoder_net.train(obs=obs_1, pred_state_mean=pred_state_1_mean, pred_state_logvar=pred_state_1_logvar, omega=current_omega)
        TENSORBOARD.loss_encoder(loss_encoder)
        TENSORBOARD.gamma(self.encoder_net.gamma)

        TENSORBOARD.write_all_metrics(step=step)

    def mcts_step_simulate(self, start_state, simulation_depth):
        """
        Starting from a state, this function uses the habitual net to predict the action taken, then using the
        starting state and the predicted action it predicts the next state using the transition net, and repeats this
        until simulation_depth is reached.
        """

        # Init empty np.arrays to store results of simulation
        # TODO: try to refactor to tf tensors so eager mode can be disabled
        start_states = np.zeros((simulation_depth, cfg.state_dim), cfg.np_precision)
        pred_states = np.zeros((simulation_depth, cfg.state_dim), cfg.np_precision)
        pred_states_mean = np.zeros((simulation_depth, cfg.state_dim), cfg.np_precision)
        pred_states_logvar = np.zeros((simulation_depth, cfg.state_dim), cfg.np_precision)
        actions = np.zeros((simulation_depth, cfg.action_dim), cfg.np_precision)

        start_states[0] = start_state

        # Loop through simulation depths and select an action using the habitual net
        for d in range(0, simulation_depth):
            try:
                # Get state of current depth and add batch dimension of 1
                state_d = start_states[d].reshape(1, -1)

                # Predict action usually taken given current state using the habitual net
                P_action = self.habitual_net.predict_action(state_d)
                P_action = tf.squeeze(P_action).numpy()

                # Choose an action from the predicted distribution and save it to the register as one-hot
                depth_d_action = np.random.choice(cfg.action_dim, p=P_action)
                actions[d, depth_d_action] = 1.0

            except Exception as e:
                print("Mysterious EXCEPTION!", e)
                # Select 'do-nothing' action
                actions[d, 0] = 1.0

            # Get state and action of current depth
            action_d_onehot = actions[d].reshape(1, -1)
            state_d = start_states[d].reshape(1, -1)

            # Predict next state given current state and predicted action (by the habitual net)
            pred_state_next, pred_state_next_mean, pred_state_next_logvar = self.transition_net.transition_with_sample(action_d_onehot, state_d)

            # Save predicted state to the trajectory register
            pred_states[d] = tf.squeeze(pred_state_next).numpy()
            pred_states_mean[d] = tf.squeeze(pred_state_next_mean).numpy()
            pred_states_logvar[d] = tf.squeeze(pred_state_next_logvar).numpy()

            # If not in last level of depth
            if d + 1 < simulation_depth:
                # Assign predicted state to trajectory register to be used as start state for next level of depth
                start_states[d + 1] = tf.squeeze(pred_state_next).numpy()

        # Calculate G given the generated trajectory for each level
        G_of_trajectory = self.calculate_G_given_trajectory(start_states, pred_states, pred_states_mean, pred_states_logvar, actions)

        # Get the mean of the levels
        G = tf.reduce_mean(G_of_trajectory).numpy()

        return G

    def save(self, save_path):
        self.habitual_net.save(save_path / "habitual_net")
        self.transition_net.save(save_path / "transition_net")
        self.encoder_net.save(save_path / "encoder_net")

    def load(self, load_path):
        self.habitual_net.load(load_path / "habitual_net")
        self.transition_net.load(load_path / "transition_net")
        self.encoder_net.load(load_path / "encoder_net")

    def create_checkpoint(self):
        # NOTE: the link below might help fixing the bug with spiking loss after restore. Idea would be to separately save optimizer weights then restore them.
        # https://stackoverflow.com/questions/49503748/save-and-load-model-optimizer-state

        self.checkpoint_manager.save()

    def resume_training(self, checkpoints_path):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoints_path)
        resume_status = self.checkpoint.restore(latest_checkpoint)

        # Instead of applying a zero-grad training step, this code is cleaner:
        for model_name in ["habitual_net", "transition_net", "encoder_net"]:
            model = getattr(self, model_name)
            with tf.name_scope(model.optimizer._name):
                with tf.init_scope():
                    model.optimizer._create_all_weights(model.trainable_variables)

        # ========= CHECKPOINT EXPLORER =========
        # reader = tf.train.load_checkpoint(checkpoints_path)
        # shape_from_key = reader.get_variable_to_shape_map()

        # Print keys in loaded checkpoint
        # print(sorted(shape_from_key.keys()))

        # Print value in one of the above retriever keys
        # print("checkpoint value", np.array(reader.get_tensor("encoder_net/.../VARIABLE_VALUE")).mean())
        # ========= /CHECKPOINT EXPLORER =========

        # Make sure that the model and checkpoint match exactly
        resume_status.assert_consumed()

        return latest_checkpoint
