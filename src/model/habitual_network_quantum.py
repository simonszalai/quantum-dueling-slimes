import os
import json
import tensorflow as tf

import src.train.config as cfg
from src.utils import stable_tf_log


class HabitualNetworkQuantum(tf.keras.Model):
    """
    Habitual network, used to reduce computational burden for states that are often visited. It predicts normally which action is
    taken given a state.
    """

    def __init__(self, add_quantum_noise):
        super(HabitualNetworkQuantum, self).__init__()
        self.add_quantum_noise = add_quantum_noise

        self.noise_path = "quantum_noise"
        self.noise_storage = {}
        self.file_cursor = 0
        self.row_cursor = 0

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rates.get("habitual"))
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=((cfg.state_dim + cfg.noise_dim,))),
                tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dense(units=512, activation=tf.nn.relu, kernel_initializer="he_uniform"),
                tf.keras.layers.Dense(cfg.action_dim),
            ]
        )  # No activation

        if self.add_quantum_noise:
            self.read_noise_files()

    def read_noise_files(self):
        """
        Loads pre-generated quantum noise from JSON files to memory.
        """
        noise_files = [f for f in os.listdir(self.noise_path) if os.path.isfile(os.path.join(self.noise_path, f))]
        self.last_noise_file = max([int(noise_file.split(".")[0]) for noise_file in noise_files])
        self.load_current_noise_file()

    def load_current_noise_file(self):
        with open(f"{self.noise_path}/{self.file_cursor}.json", "r") as f:
            noise_content = json.load(f)
            self.noise_storage[self.file_cursor] = noise_content

    def get_quantum_noise(self):
        # Get the loaded content of the current file
        file_content = self.noise_storage[self.file_cursor]

        # Check if the current row cursor points to a row that exists
        if self.row_cursor > len(file_content) - 1:
            # If not, move to the next file and reset row cursor
            self.row_cursor = 0
            self.file_cursor += 1
            if self.file_cursor > self.last_noise_file:
                raise Exception("Ran out of noise files, stopping.")

            self.load_current_noise_file()

        # Get noise and increment row cursor
        self.row_cursor += 1
        return self.noise_storage[self.file_cursor][self.row_cursor]

    def predict_action(self, state):
        if self.add_quantum_noise:
            noise = self.get_quantum_noise()
        else:
            noise = tf.zeros(cfg.noise_dim)

        # Raw outputs of the neural network
        noise = tf.convert_to_tensor(noise)
        noise = tf.cast(noise, cfg.tf_precision)
        noise = tf.expand_dims(noise, axis=0)
        logits_action = self.model(tf.concat([state, noise], axis=1))

        # Probability of each action selected given a state
        # Q denotes a probability distribution just like P, but a different letter is used to amplify that it is different from the other
        # probability distribution P which is calculated from Expected Free Energy
        P_action = tf.nn.softmax(logits_action)

        return P_action

    @tf.function
    def compute_loss(self, state, P_action_internal):
        """
        The habitual network learns to imitate the action selections of the agent's internal model,
        so it can be used as a perf optimization

        Parameters
        ----------
        state : np.array
            Observation encoded as a lower dimensional state by the encoder network
        P_action_internal: np.array
            Probability distribution of actions predicted by the agent's internal model (encoder + transition networks)
            or any other P_action that the habitual network should train to replicate (e.g. slimevolley baseline policy)
        """

        # Probability distribution of actions predicted by the habitual network
        P_action_habitual = self.predict_action(state)

        # Calculate Kullback-Leibler Divergence between the output of the habitual network and the agent's internal model
        D_KL_action = P_action_habitual * (stable_tf_log(P_action_habitual) - stable_tf_log(P_action_internal))

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
            loss = self.compute_loss(state=tf.stop_gradient(state), P_action_internal=tf.stop_gradient(P_action))
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    def save(self, checkpoint_path):
        self.model.save(checkpoint_path)

    def load(self, checkpoint_path):
        self.model = tf.keras.models.load_model(checkpoint_path)
