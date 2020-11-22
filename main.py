from neuraln import DistortionANN

def main():
	ann = DistortionANN(data_path='./dataset')
	ann.train(1)
	ann.predict('tests/riff.wav')

if __name__ == '__main__':
	main()
