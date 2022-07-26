import tensorflow as tf

import src.utils as utils
import src.train.config as cfg


class EncoderNetwork(tf.keras.Model):
    """
    A Variational Autoencoder to encode and decode state from/to observations
    Short explanation: https://github.com/bvezilic/Variational-autoencoder
    Longer explanation: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73

    """

    def __init__(self):
        super(EncoderNetwork, self).__init__()

        self.beta_state = tf.Variable(cfg.beta_state, trainable=False, name="beta_state")
        self.beta_obs = tf.Variable(cfg.beta_obs, trainable=False, name="beta_obs")
        self.gamma = tf.Variable(cfg.gamma, trainable=False, name="gamma")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rates.get("encoder"))
        self.encoder_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(cfg.state_dim)),
                tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(cfg.state_dim + cfg.state_dim),
            ]
        )  # No activation

        self.decoder_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(cfg.state_dim,)),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(cfg.state_dim, activation="sigmoid", kernel_initializer="he_uniform"),
            ]
        )

    @tf.function
    def encode(self, obs):
        """
        Encode observation as low-dimensional state, then reparameterize
        logvar: log variance

        Returns
        -------
        state : np.array
        """

        network_out = self.encoder_model(obs)
        state_mean, state_logvar = tf.split(network_out, num_or_size_splits=2, axis=1)
        state = utils.reparameterize(state_mean, state_logvar)
        return state, state_mean, state_logvar

    @tf.function
    def decode(self, state):
        pred_obs = self.decoder_model(state)
        return pred_obs

    @tf.function
    def compute_loss(self, obs, pred_state_mean, pred_state_logvar, omega):
        # Encode the actual observation TODO: possible optimization, reuse obs -> state computed in active_inference.py:292
        actual_state_1, actual_state_1_mean, actual_state_1_logvar = self.encode(obs)

        # Decode the encoded state back to observation
        pred_obs_1 = self.decode(actual_state_1)

        # TERM: Eq[log P(o1|s1)] -> Binary Cross Entropy
        bin_cross_entr = obs * utils.stable_tf_log(pred_obs_1) + (1 - obs) * utils.stable_tf_log(1 - pred_obs_1)
        log_pred_obs_1_state_1 = tf.reduce_sum(bin_cross_entr, axis=[1])

        # TERM: Eqpi D_kl[Q(s1)||N(0.0,1.0)]
        D_KL_naive = utils.D_KL_from_logvar_and_precision(actual_state_1_mean, actual_state_1_logvar, 0.0, 0.0, omega)

        # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
        D_KL = utils.D_KL_from_logvar_and_precision(actual_state_1_mean, actual_state_1_logvar, pred_state_mean, pred_state_logvar, omega)

        # Beginning of training (only use D_KL_naive)
        if self.gamma <= 0.05:
            loss = -self.beta_obs * log_pred_obs_1_state_1 + self.beta_state * D_KL_naive
        # End of training (only use D_KL, with current config, it never gets here)
        elif self.gamma >= 0.95:
            loss = -self.beta_obs * log_pred_obs_1_state_1 + self.beta_state * D_KL
        # Middle of training (use a mixture of D_KL and D_KL_naive)
        else:
            loss = -self.beta_obs * log_pred_obs_1_state_1 + self.beta_state * (self.gamma * D_KL + (1.0 - self.gamma) * D_KL_naive)

        return loss

    @tf.function
    def train(self, obs, pred_state_mean, pred_state_logvar, omega):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(obs, tf.stop_gradient(pred_state_mean), tf.stop_gradient(pred_state_logvar), omega=tf.stop_gradient(omega))
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def save(self, checkpoint_path):
        self.encoder_model.save(checkpoint_path / "encoder")
        self.decoder_model.save(checkpoint_path / "decoder")

    def load(self, checkpoint_path):
        self.encoder_model = tf.keras.models.load_model(checkpoint_path / "encoder")
        self.decoder_model = tf.keras.models.load_model(checkpoint_path / "decoder")
