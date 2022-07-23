import tensorflow as tf

import src.train.config as cfg
from src.utils import D_KL_from_logvar_and_precision, stable_tf_log


class EncoderNetwork(tf.keras.Model):
    """
    Network to encode and decode state from/to observations
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
                tf.keras.layers.Dense(16, activation=tf.keras.activations.linear, kernel_initializer="identity"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(cfg.state_dim + cfg.state_dim, activation=tf.keras.activations.linear, kernel_initializer="identity"),
            ],
            name="encoder_sequential",
        )  # No activation

        self.decoder_model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(cfg.state_dim,)),
                tf.keras.layers.Dense(16, activation=tf.keras.activations.linear, kernel_initializer="identity"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(cfg.state_dim, activation=tf.keras.activations.linear, kernel_initializer="identity"),
            ],
            name="decoder_sequential",
        )

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    @tf.function
    def encode(self, obs):
        """
        Encode observation as low-dimensional state
        """

        network_out = self.encoder_model(obs)
        mean_state, logvar_state = tf.split(network_out, num_or_size_splits=2, axis=1)
        return mean_state, logvar_state

    @tf.function
    def decode(self, state):
        pred_obs = self.decoder_model(state)
        return pred_obs

    @tf.function
    def encode_with_sample(self, obs):
        """
        Encode observation as low-dimensional state, then reparameterize
        """
        mean, logvar = self.encode(obs)
        state = self.reparameterize(mean, logvar)
        return state, mean, logvar

    @tf.function
    def compute_loss(self, obs_1, pred_state_1_mean, pred_state_1_logvar, omega):
        # Encode the actual observation
        actual_state_1, actual_state_1_mean, actual_state_1_logvar = self.encode_with_sample(obs_1)

        # Decode the encoded state back to observation
        pred_obs_1 = self.decode(actual_state_1)

        # TERM: Eq[log P(o1|s1)]
        bin_cross_entr = obs_1 * stable_tf_log(pred_obs_1) + (1 - obs_1) * stable_tf_log(1 - pred_obs_1)  # Binary Cross Entropy
        log_pred_obs_1_state_1 = tf.reduce_sum(bin_cross_entr, axis=[1])

        # TERM: Eqpi D_kl[Q(s1)||N(0.0,1.0)]
        D_KL_naive = D_KL_from_logvar_and_precision(actual_state_1_mean, actual_state_1_logvar, 0.0, 0.0, omega)

        # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
        D_KL = D_KL_from_logvar_and_precision(actual_state_1_mean, actual_state_1_logvar, pred_state_1_mean, pred_state_1_logvar, omega)

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
    def train(self, obs_1, pred_state_1_mean, pred_state_1_logvar, omega):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(obs_1, tf.stop_gradient(pred_state_1_mean), tf.stop_gradient(pred_state_1_logvar), omega=tf.stop_gradient(omega))
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def save(self, checkpoint_path):
        self.encoder_model.save(checkpoint_path / "encoder")
        self.decoder_model.save(checkpoint_path / "decoder")

    def load(self, checkpoint_path):
        self.encoder_model = tf.keras.models.load_model(checkpoint_path / "encoder")
        self.decoder_model = tf.keras.models.load_model(checkpoint_path / "decoder")
