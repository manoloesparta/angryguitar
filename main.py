from angry import AngryNeuralNetwork 

def main():
	ann = AngryNeuralNetwork(data_path='./dataset')
	ann.train(10)
	ann.predict_clip('tests/solo.wav')

if __name__ == '__main__':
	main()
