import tensorflow as tf


class AutoEncoder(tf.keras.Model):

	def __init__(self):
		super(AutoEncoder, self).__init__()

		self.encoder = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(128, 9, 1)), 
			tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
			tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2)
		])

		self.decoder = tf.keras.Sequential([
			tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
			tf.keras.layers.Conv2D(1, kernel_size=(3,3), activation='sigmoid', padding='same')
		])

	def call(self, x):
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded