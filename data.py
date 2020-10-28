import os
import librosa
import numpy as np
import librosa.display
from pydub import AudioSegment
from pydub.playback import play
import matplotlib.pyplot as plt


def create_audio_clips():
    clean = AudioSegment.from_file('clean.wav', 'wav') 
    distor = AudioSegment.from_file('distorted.wav', 'wav')

    minimum = min((clean), len(distor))
    
    for i in range(0, minimum - 200, 200):
        clean_slice = clean[i:i+200]
        clean_slice.export(f'dataset/clean_wav/clean{i//200}.wav',format='wav')

        distor_slice = distor[i:i+200]
        distor_slice.export(f'dataset/distorted_wav/distor{i//200}.wav', format='wav')


def create_mucho_texto():
    clean_wav = os.listdir('dataset/clean_wav/')
    distor_wav = os.listdir('dataset/distorted_wav/')

    for i, j in zip(clean_wav, distor_wav):
        clean_y, clean_sr = librosa.load(f'dataset/clean_wav/{i}')
        clean_arr = librosa.feature.melspectrogram(y=clean_y, sr=clean_sr)
        np.savetxt(f'dataset/clean_txt/{i}.txt', clean_arr, fmt='%10.20f')

        distor_y, distor_sr = librosa.load(f'dataset/distorted_wav/{j}')
        distor_arr = librosa.feature.melspectrogram(y=distor_y, sr=distor_sr)
        np.savetxt(f'dataset/distorted_txt/{j}.txt', distor_arr, fmt='%10.20f')


if __name__ == "__main__": 
    create_mucho_texto()
