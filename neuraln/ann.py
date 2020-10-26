from . import load
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

class NeuralNetwork():


	def __init__(self, data_path):

		self.model = keras.Sequential([
			# Input
			keras.layers.Flatten(input_shape=(128,9)),

			# Hidden
			keras.layers.Dense(2048, activation='relu'),
			keras.layers.Dense(2048, activation='relu'),
			keras.layers.Dense(2048, activation='relu'),
			keras.layers.Dense(2048, activation='relu'),
			keras.layers.Dense(2048, activation='relu'),
			keras.layers.Dense(2048, activation='relu'),

			# Output
			keras.layers.Dense(128*9)
		])
		self.model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

		x, y = load.load_dataset(data_path)
		self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1)


	def train(self, epochs=10):
		flat_y = np.array([i.flatten() for i in self.y_train])
		self.model.fit(self.x_train, flat_y, epochs=epochs)


	def predict(self, file_path):
		pass


	@staticmethod
	def from_saved():
		pass
