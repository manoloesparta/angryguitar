from angry import AngryNeuralNetwork 

def main():
	ann = AngryNeuralNetwork(data_path='./dataset')
	ann.train(10)
	ann.predict_clip('tests/riff.wav')
	input("hola termine uno")
	ann.predict_clip('tests/solo.wav')
	input("hola termine otro")
	ann.predict_clip('tests/power.wav')
	# ann.predict_sample('tmp/audio0.wav', 0)

if __name__ == '__main__':
	main()
