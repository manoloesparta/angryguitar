import tensorflow as tf
from tensorflow.keras import layers

class NeuralNetwork(tf.keras.Model):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.model = tf.keras.Sequential([
            # Input
            layers.Flatten(),

            # Hidden
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1200, activation='relu'),

            # Output
            layers.Dense(50*9, activation='relu'),
            layers.Reshape((50,9))
        ])


    def call(self, x):
        return self.model(x)
