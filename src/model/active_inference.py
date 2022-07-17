import numpy as np
import tensorflow as tf

from src.train.config import action_dim, np_precision, calc_G_steps_ahead, average_G_over_N_samples, omega_params
from src.model.habitual_network import HabitualNetwork
from src.model.transition_network import TransitionNetwork
from src.model.encoder_network import EncoderNetwork
from src.utils import entropy_bernoulli, entropy_normal_from_logvar, softmax_multi_with_log, action_to_multi_hot, compute_omega
from src.train.metrics import TENSORBOARD


class ActiveInferenceModel:
    def __init__(self, state_dim, action_dim, gamma, beta_state, beta_obs, learning_rates={}, training_run_path=None, np_precision=np.float32):
        self.np_precision = np_precision
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.omega = tf.Variable(1.0, trainable=False, name="omega")

        self.tf_precision = f"float{np.finfo(np_precision).bits}"
        tf.keras.backend.set_floatx(self.tf_precision)

        self.habitual_net = HabitualNetwork(state_dim, action_dim, learning_rates.get("habitual"))
        self.transition_net = TransitionNetwork(state_dim, action_dim, learning_rates.get("transition"))
        self.encoder_net = EncoderNetwork(state_dim, beta_state, beta_obs, gamma, learning_rates.get("encoder"))

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

        self.pi_one_hot = tf.Variable(np.eye(4), trainable=False, dtype=self.tf_precision)
        self.pi_one_hot_3 = tf.Variable(np.eye(3), trainable=False, dtype=self.tf_precision)

    def check_reward(self, obs_for_actions):
        """
        Calculate reward for each action
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

        results = []
        for obs in obs_for_actions:
            x_agent, y_agent, Vx_agent, Vy_agent, x_ball, y_ball, Vx_ball, Vy_ball, X_opponent, Y_opponent, Vx_opponent, Vy_opponent = obs.numpy()
            # x_ball = tf.gather(obs, 4)
            # y_ball = tf.gather(obs, 5)
            # Vx_ball = tf.gather(obs, 6)
            # Vy_ball = tf.gather(obs, 7)

            reward = x_ball + Vx_ball
            if x_ball > 0:
                reward += -1 * (y_ball + Vy_ball)
            else:
                reward += y_ball + Vy_ball
            results.append(reward)

        return results

    @tf.function
    def habitual_network(self, obs):
        pred_state_mean, _ = self.encoder_net.encode(obs)
        Q_action = self.habitual_net.predict_action(pred_state_mean)
        return Q_action

    @tf.function
    def calculate_G_repeated(self, obs, action, steps=1, calc_mean=False, average_G_over_N_samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously.
        """
        # Calculate current s_t
        encoded_state_0_mean, encoded_state_0_logvar = self.encoder_net.encode(obs)
        encoded_state_0 = self.encoder_net.reparameterize(encoded_state_0_mean, encoded_state_0_logvar)

        sum_G = tf.zeros([obs.shape[0]], self.np_precision)

        # Predict s_t+1 for various policies
        if calc_mean:
            s0_temp = encoded_state_0_mean
        else:
            s0_temp = encoded_state_0

        for t in range(steps):
            G, pred_state_1, pred_state_1_mean = self.calculate_G(s0_temp, action, average_G_over_N_samples=average_G_over_N_samples)

            sum_G += G

            if calc_mean:
                s0_temp = pred_state_1_mean
            else:
                s0_temp = pred_state_1

        return sum_G

    @tf.function
    def calculate_G(self, state_0, action_0, average_G_over_N_samples=10):
        """
        Calculate Expected Free Energy
        """
        term0 = tf.zeros([state_0.shape[0]], self.np_precision)
        term1 = tf.zeros([state_0.shape[0]], self.np_precision)
        term2_1 = tf.zeros(state_0.shape[0], self.np_precision)
        term2_2 = tf.zeros(state_0.shape[0], self.np_precision)

        for _ in range(average_G_over_N_samples):
            pred_state_1, pred_state_1_mean, pred_state_1_logvar = self.transition_net.transition_with_sample(state_0, action_0)
            pred_obs_1 = self.encoder_net.decode(pred_state_1)
            _, _, encoded_pred_state_1_logvar = self.encoder_net.encode_with_sample(pred_obs_1)

            # E [ log P(o|pi) ]
            log_pred_obs_1 = self.check_reward(pred_obs_1)
            term0 += log_pred_obs_1

            # E [ log Q(s|pi) - log Q(s|o,pi) ]
            term1_new = -tf.reduce_sum(entropy_normal_from_logvar(pred_state_1_logvar) + entropy_normal_from_logvar(encoded_pred_state_1_logvar), axis=1)
            term1 += term1_new

            # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
            pred_state_1_temp1, _, _ = self.transition_net.transition_with_sample(state_0, action_0)
            pred_obs_1_temp1 = self.encoder_net.decode(pred_state_1_temp1)
            term2_1_new = tf.reduce_sum(entropy_bernoulli(pred_obs_1_temp1), axis=[1])
            term2_1 += term2_1_new

            # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
            pred_obs_temp2 = self.encoder_net.decode(pred_state_1)
            term2_2_new = tf.reduce_sum(entropy_bernoulli(pred_obs_temp2), axis=[1])
            term2_2 += term2_2_new

        term0 /= float(average_G_over_N_samples)
        term1 /= float(average_G_over_N_samples)
        term2_1 /= float(average_G_over_N_samples)
        term2_2 /= float(average_G_over_N_samples)

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2

        return G, pred_state_1, pred_state_1_mean

    # TODO: refactor
    @tf.function
    def calculate_G_mean(self, s0, pi0):

        _, ps1_mean, ps1_logvar = self.transition_net.transition_with_sample(pi0, s0)
        po1 = self.encoder_net.decode(ps1_mean)
        _, qs1_mean, qs1_logvar = self.encoder_net.encode_with_sample(po1)

        # E [ log P(o|pi) ]
        logpo1 = self.check_reward(po1)
        term0 = logpo1

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = -tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar) + entropy_normal_from_logvar(qs1_logvar), axis=1)

        # Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.encoder_net.decode(self.transition_net.transition_with_sample(pi0, s0)[1])
        term2_1 = tf.reduce_sum(entropy_bernoulli(po1_temp1), axis=[1, 2, 3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.encoder_net.decode(self.encoder_net.reparameterize(ps1_mean, ps1_logvar))
        term2_2 = tf.reduce_sum(entropy_bernoulli(po1_temp2), axis=[1, 2, 3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        G = -term0 + term1 + term2

        return G, [term0, term1, term2], ps1_mean, po1

    # TODO: refactor
    @tf.function
    def calculate_G_4_repeated(self, o, steps=1, calc_mean=False, samples=10):
        """
        We simultaneously calculate G for the four policies of repeating each
        one of the four actions continuously..
        """
        # Calculate current s_t
        qs0_mean, qs0_logvar = self.encoder_net.encode(o)
        qs0 = self.encoder_net.reparameterize(qs0_mean, qs0_logvar)

        sum_terms = [tf.zeros([4], self.tf_precision), tf.zeros([4], self.tf_precision), tf.zeros([4], self.tf_precision)]
        sum_G = tf.zeros([4], self.tf_precision)

        # Predict s_t+1 for various policies
        if calc_mean:
            s0_temp = qs0_mean
        else:
            s0_temp = qs0

        for t in range(steps):
            if calc_mean:
                G, terms, ps1_mean, po1 = self.calculate_G_mean(s0_temp, self.pi_one_hot)
            else:
                G, terms, s1, ps1_mean, po1 = self.calculate_G(s0_temp, self.pi_one_hot, samples=samples)

            sum_terms[0] += terms[0]
            sum_terms[1] += terms[1]
            sum_terms[2] += terms[2]
            sum_G += G

            if calc_mean:
                s0_temp = ps1_mean
            else:
                s0_temp = s1

        return sum_G, sum_terms, po1

    @tf.function
    def calculate_G_given_trajectory(self, s0_traj, ps1_traj, ps1_mean_traj, ps1_logvar_traj, pi0_traj):
        # NOTE: len(s0_traj) = len(s1_traj) = len(pi0_traj)

        po1 = self.encoder_net.decode(ps1_traj)
        qs1, _, qs1_logvar = self.encoder_net.encode_with_sample(po1)

        # E [ log P(o|pi) ]
        term0 = self.check_reward(po1)

        # E [ log Q(s|pi) - log Q(s|o,pi) ]
        term1 = -tf.reduce_sum(entropy_normal_from_logvar(ps1_logvar_traj) + entropy_normal_from_logvar(qs1_logvar), axis=1)

        #  Term 2.1: Sampling different thetas, i.e. sampling different ps_mean/logvar with dropout!
        po1_temp1 = self.encoder_net.decode(self.transition_net.transition_with_sample(pi0_traj, s0_traj)[0])
        term2_1 = tf.reduce_sum(entropy_bernoulli(po1_temp1), axis=[1, 2, 3])

        # Term 2.2: Sampling different s with the same theta, i.e. just the reparametrization trick!
        po1_temp2 = self.encoder_net.decode(self.transition_net.reparameterize(ps1_mean_traj, ps1_logvar_traj))
        term2_2 = tf.reduce_sum(entropy_bernoulli(po1_temp2), axis=[1, 2, 3])

        # E [ log [ H(o|s,th,pi) ] - E [ H(o|s,pi) ]
        term2 = term2_1 - term2_2

        return -term0 + term1 + term2

    def predict_agent_action(self, obs):
        # TRSTEP 3 Run planner and compute prior policy P (at)
        # TRSTEP 3.a Define shape of action space
        dummy_action_onehot = tf.eye(action_dim, dtype=np_precision)  # Shape: (action_counts, action_counts), e.g. (3, 3)

        # TRSTEP 3.b Compute Expected Free Energy
        # samples: average G over N samples
        o0_repeated = obs.repeat(action_dim, 0)

        sum_G = self.calculate_G_repeated(
            o0_repeated, dummy_action_onehot, steps=calc_G_steps_ahead, average_G_over_N_samples=average_G_over_N_samples, calc_mean=True
        )  # Shape (batch * action_counts,), e.g. (3,)
        # TRSTEP 3.c Compute prior policy (probability distribution over actions)
        P_action, _ = softmax_multi_with_log(-sum_G.numpy(), action_dim)  # Shape: (batch, action_dim), e.g. (1, 3)

        # TRSTEP 9 Apply action a ̃ ∼ P (a ) to the environment.
        # TRSTEP 9.a Sample prior policy (action probability distributions)
        action_index = np.random.choice(action_dim, p=P_action.squeeze(axis=0))

        # Convert action choices to multi-hot (for environment)
        action_agent = action_to_multi_hot(action_index, dtype=self.tf_precision)

        # Convert action index to one-hot (for network training)
        agent_action_onehot = np.zeros((1, action_dim), dtype=np_precision)
        agent_action_onehot[0, action_index] = 1.0

        return action_agent, [action_index, P_action, agent_action_onehot]

    def train(self, obs_agent, train_info, step):
        _, P_action, agent_action_onehot = train_info

        # -- TRAIN HABITUAL NETWORK ---------------------------------------------------
        # 4. Compute Qφs (st) using o ̃t.
        state_0, _, _ = self.encoder_net.encode_with_sample(obs_agent)

        loss_habitual = self.habitual_net.train(state_0, P_action)
        TENSORBOARD.loss_habitual(loss_habitual)

        # -- TRAIN TRANSITION NETWORK ------------------------------------------------
        # 11. Compute Qφs (st+1 ) using o ̃t+1 .
        state_1_mean, state_1_logvar = self.encoder_net.encode(obs_agent)
        loss_transition, pred_state_1_mean, pred_state_1_logvar = self.transition_net.train(
            state_0, agent_action_onehot, state_1_mean=state_1_mean, state_1_logvar=state_1_logvar, omega=self.omega
        )
        TENSORBOARD.loss_transition(loss_transition)

        current_omega = compute_omega(loss_habitual, omega_params=omega_params).reshape(-1, 1)
        self.omega.assign(tf.reduce_mean(current_omega))
        TENSORBOARD.omega(self.omega)

        # -- TRAIN ENCODER NETWORK --------------------------------------------------
        loss_encoder = self.encoder_net.train(
            obs_1=obs_agent, pred_state_1_mean=pred_state_1_mean, pred_state_1_logvar=pred_state_1_logvar, omega=current_omega
        )
        TENSORBOARD.loss_encoder(loss_encoder)
        TENSORBOARD.gamma(self.encoder_net.gamma)

        TENSORBOARD.write_all_metrics(step=step)

    # TODO: refactor
    def mcts_step_simulate(self, starting_s, depth, use_means=False):
        s0 = np.zeros((depth, self.state_dim), self.np_precision)
        ps1 = np.zeros((depth, self.state_dim), self.np_precision)
        ps1_mean = np.zeros((depth, self.state_dim), self.np_precision)
        ps1_logvar = np.zeros((depth, self.state_dim), self.np_precision)
        pi0 = np.zeros((depth, self.action_dim), self.np_precision)

        s0[0] = starting_s
        try:
            Qpi_t_to_return = self.habitual_net.predict_action(s0[0].reshape(1, -1))[1].numpy()[0]
            pi0[0, np.random.choice(self.action_dim, p=Qpi_t_to_return)] = 1.0
        except Exception:
            pi0[0, 0] = 1.0
            Qpi_t_to_return = pi0[0]
        ps1_new, ps1_mean_new, ps1_logvar_new = self.transition_net.transition_with_sample(pi0[0].reshape(1, -1), s0[0].reshape(1, -1))
        ps1[0] = ps1_new[0].numpy()
        ps1_mean[0] = ps1_mean_new[0].numpy()
        ps1_logvar[0] = ps1_logvar_new[0].numpy()
        if 1 < depth:
            if use_means:
                s0[1] = ps1_mean_new[0].numpy()
            else:
                s0[1] = ps1_new[0].numpy()
        for t in range(1, depth):
            try:
                pi0[t, np.random.choice(self.action_dim, p=self.habitual_net.predict_action(s0[t].reshape(1, -1))[1].numpy()[0])] = 1.0
            except Exception:
                pi0[t, 0] = 1.0
            ps1_new, ps1_mean_new, ps1_logvar_new = self.transition_net.transition_with_sample(pi0[t].reshape(1, -1), s0[t].reshape(1, -1))
            ps1[t] = ps1_new[0].numpy()
            ps1_mean[t] = ps1_mean_new[0].numpy()
            ps1_logvar[t] = ps1_logvar_new[0].numpy()
            if t + 1 < depth:
                if use_means:
                    s0[t + 1] = ps1_mean_new[0].numpy()
                else:
                    s0[t + 1] = ps1_new[0].numpy()

        G = tf.reduce_mean(self.calculate_G_given_trajectory(s0, ps1, ps1_mean, ps1_logvar, pi0)).numpy()
        return G, pi0, Qpi_t_to_return

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
