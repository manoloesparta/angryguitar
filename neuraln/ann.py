import numpy as np
from . import load
from tqdm import tqdm
import librosa.display
import tensorflow as tf
from . import AutoEncoder
import os, shutil, librosa
from tensorflow import keras
from pydub import AudioSegment
from pydub.playback import play
from sklearn.model_selection import train_test_split


class DistortionANN():

	SAMPLE_RATE = 44100

	def __init__(self, data_path):
		self.data_path = data_path
		self.model = AutoEncoder()
		self.model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy'])


	def train(self, epochs=10):
		x, y = load.load_dataset(self.data_path)
		x = x[..., tf.newaxis]
		y = y[..., tf.newaxis]

		print(x.shape)
		print(y.shape)

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

		self.model.fit(x_train, y_train, epochs=epochs)
		test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=1) 

		print('Test accuracy:', test_acc)


	def predict(self, file_path, output='output', remove=False):
		if os.path.exists('tmp'):
			shutil.rmtree('tmp')
		os.mkdir('tmp')

		arr = []
		audio = AudioSegment.from_file(file_path, 'wav')

		for i in range(0, len(audio) - 200, 200):
			audio_slice = audio[i:i+200]
			audio_slice.export(f'tmp/audio{i//200}.wav',format='wav')
			y, sr = librosa.load(f'tmp/audio{i//200}.wav')
			arr.append(librosa.feature.melspectrogram(y=y, sr=sr))

		res = self.model.predict(np.array(arr))
		shutil.rmtree('tmp')
		os.mkdir('tmp')

		for i, val in enumerate(tqdm(res)):
			audio = librosa.feature.inverse.mel_to_audio(val)
			librosa.output.write_wav(f'tmp/{output}{i}.wav', audio, self.SAMPLE_RATE//2)

		final = AudioSegment.empty()
		for i, _ in enumerate(res):
			final += AudioSegment.from_wav(f'tmp/{output}{i}.wav')

		final.export(f'{output}.wav', format='wav')

		if remove:
			shutil.rmtree('tmp')
