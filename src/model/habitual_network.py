import tensorflow as tf

import src.train.config as cfg
from src.utils import stable_tf_log


class HabitualNetwork(tf.keras.Model):
    """
    Habitual network, used to reduce computational burden for states that are often visited. It predicts normally which action is
    taken given a state.
    """

    def __init__(self):
        super(HabitualNetwork, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rates.get("habitual"))
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(cfg.state_dim,)),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dense(units=128, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dense(cfg.action_dim),
            ]
        )  # No activation

    def predict_action(self, state):
        # Raw outputs of the neural network
        logits_action = self.model(state)

        # Probability of each action selected given a state
        # Q denotes a probability distribution just like P, but a different letter is used to amplify that it is different from the other
        # probability distribution P which is calculated from Expected Free Energy
        Q_action = tf.nn.softmax(logits_action)

        return Q_action

    @tf.function
    def compute_loss(self, state, P_action):
        """
        Parameters
        ----------
        state : np.array
            Observation encoded as a lower dimensional state by the encoder network
        log_P_action: np.array
            Probabilities of each action to be selected based on calculated Expected Free Energy
        """

        # Q_action: probability of each action selected given a state
        Q_action = self.predict_action(state)

        # Calculate Kullback-Leibler Divergence
        D_KL_action = Q_action * (stable_tf_log(Q_action) - stable_tf_log(P_action))

        # Sum up KL divergence to get loss
        loss = tf.reduce_sum(D_KL_action, 1)

        return loss

    @tf.function
    def train(self, state, P_action):
        """
        Parameters
        ----------
        state : np.array
            Observation encoded as a lower dimensional state by the encoder network
        log_P_action: np.array
            Probabilities of each action to be selected
        """

        with tf.GradientTape() as tape:
            loss = self.compute_loss(state=tf.stop_gradient(state), P_action=tf.stop_gradient(P_action))
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def save(self, checkpoint_path):
        self.model.save(checkpoint_path)

    def load(self, checkpoint_path):
        self.model = tf.keras.models.load_model(checkpoint_path)
