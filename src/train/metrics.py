import tensorflow as tf
from pathlib import Path


class Tensorboard:
    def __init__(self):
        # Define a list of metric names
        self.metrics = ["loss_encoder", "loss_habitual", "loss_transition", "omega", "gamma"]

        # Create a TensorFlow metric for each element of the above list (these are to be called during training with the actual values)
        for metric in self.metrics:
            # self[metric] = tf.keras.metrics.Mean(name=metric)
            setattr(Tensorboard, metric, tf.keras.metrics.Mean(name=metric))

        # self.loss_encoder = tf.keras.metrics.Mean(name="loss_encoder")

    def create_writer(self, training_run_path):
        # Define a writer that specifies the folder where logs should be saved
        train_log_dir = Path(training_run_path, "logs")
        self.writer = tf.summary.create_file_writer(train_log_dir.as_posix())

    def write_all_metrics(self, step):
        # Let metrics stabilize in the beginning
        # if step < 50:
        #     return

        with self.writer.as_default():
            for metric in self.metrics:
                tf.summary.scalar(metric, getattr(self, metric).result(), step=step)


TENSORBOARD = Tensorboard()
