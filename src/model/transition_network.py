import tensorflow as tf

import src.train.config as cfg
from src.utils import D_KL_from_logvar_and_precision


class TransitionNetwork(tf.keras.Model):
    """
    Transition function (B)
    """

    def __init__(self):
        super(TransitionNetwork, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rates.get("transition"))
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(cfg.action_dim + cfg.state_dim,), name="transition_input"),
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu, kernel_initializer="he_uniform", name="transition_dense_1"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu, kernel_initializer="he_uniform", name="transition_dense_2"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(units=32, activation=tf.nn.relu, kernel_initializer="he_uniform", name="transition_dense_3"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(cfg.state_dim + cfg.state_dim),
            ],
            name="transition_sequential",
        )  # No activation

    @tf.function
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    @tf.function
    def transition(self, action, state):
        # Expand dimensions so state and action can be concatenated
        try:
            action.shape[1]
        except IndexError:
            action = tf.expand_dims(action, axis=0)

        try:
            state.shape[1]
        except IndexError:
            state = tf.expand_dims(state, axis=0)

        net_input = tf.concat([action, state], axis=1)
        net_output = self.model(net_input)
        mean, logvar = tf.split(net_output, num_or_size_splits=2, axis=1)
        return mean, logvar

    @tf.function
    def transition_with_sample(self, state, action):
        # Remove empty dimensions, so state and action can be concatenated
        try:
            for axis in range(2):
                state = tf.squeeze(state, axis=axis)

        # If state is tensorflow.python.framework.ops.EagerTensor, tf.errors.InvalidArgumentError is thrown
        # If state is tensorflow.python.framework.ops.Tensor, ValueError is thrown
        except (tf.errors.InvalidArgumentError, ValueError):
            pass

        try:
            for axis in range(2):
                action = tf.squeeze(action, axis=axis)
        except (tf.errors.InvalidArgumentError, ValueError):
            pass

        pred_state_1_mean, pred_state_1_logvar = self.transition(action, state)
        pred_state_1 = self.reparameterize(pred_state_1_mean, pred_state_1_logvar)
        return pred_state_1, pred_state_1_mean, pred_state_1_logvar

    @tf.function
    def compute_loss(self, pred_state_1_mean, pred_state_1_logvar, actual_state_1_mean, actual_state_1_logvar, omega):
        # TERM: Eqpi D_kl[Q(s1)||P(s1|s0,pi)]
        loss = D_KL_from_logvar_and_precision(actual_state_1_mean, actual_state_1_logvar, pred_state_1_mean, pred_state_1_logvar, omega)
        return loss

    @tf.function
    def train(self, state_0, action_0, state_1_mean, state_1_logvar, omega):
        """
        Parameters
        ----------
        state_1_mean, state_1_logvar
            Mean and logvar of the actual next state, encoded from the observation returned by the environment.
        """
        with tf.GradientTape() as tape:
            # Predict the next state from current state and applied action (use the agent's model of the environment)
            _, pred_state_1_mean, pred_state_1_logvar = self.transition_with_sample(tf.stop_gradient(state_0), tf.stop_gradient(action_0))

            loss = self.compute_loss(
                pred_state_1_mean=pred_state_1_mean,
                pred_state_1_logvar=pred_state_1_logvar,
                actual_state_1_mean=tf.stop_gradient(state_1_mean),
                actual_state_1_logvar=tf.stop_gradient(state_1_logvar),
                omega=tf.stop_gradient(omega),
            )
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss, pred_state_1_mean, pred_state_1_logvar

    def save(self, checkpoint_path):
        self.model.save(checkpoint_path)

    def load(self, checkpoint_path):
        self.model = tf.keras.models.load_model(checkpoint_path)
