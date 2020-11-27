import os
import numpy as np
from tqdm import tqdm

def load_dataset(path):
	files_count = len(os.listdir(f'{path}/clean_txt/'))
	print(files_count)

	clean_data = []
	distor_data = []

	print('Loading data...')
	for i in tqdm(range(files_count)):

		clean_tmp = np.loadtxt(f'{path}/clean_txt/clean{i}.wav.txt', dtype=np.float64)
		clean_data.append(clean_tmp)

		distor_tmp = np.loadtxt(f'{path}/distorted_txt/distor{i}.wav.txt', dtype=np.float64)
		distor_data.append(distor_tmp)

	clean_data = np.array(clean_data)
	distor_data = np.array(distor_data)

	print(clean_data.shape)

	return clean_data, distor_data