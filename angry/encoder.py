import tensorflow as tf
from tensorflow.keras import layers

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1200, activation='relu'),
            layers.Dense(1800, activation='relu'),
            layers.Dense(1800, activation='relu'),
            layers.Dense(1800, activation='relu'),
            layers.Dense(1200, activation='relu'),
            layers.Dense(128*9, activation='relu'),
            layers.Reshape((128, 9))
        ])


    def call(self, x):
        encoded = self.encoder(x)
        return encoded 
