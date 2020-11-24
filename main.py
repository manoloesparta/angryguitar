from angry import AngryNeuralNetwork 

def main():
	ann = AngryNeuralNetwork(data_path='./dataset')
	ann.train(100)
	ann.predict('tests/riff.wav', remove=True)

if __name__ == '__main__':
	main()
