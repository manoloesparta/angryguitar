import os
from neuraln import NeuralNetwork

def main():
	ann = NeuralNetwork(data_path='./dataset')
	ann.train(1)
	ann.predict('tests/riff.wav')

if __name__ == '__main__':
	main()