from Caculate import extract_features as ef
import CONFIG as c
import pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture
from recordAudio import record_audio
from test import test_list_file

def train_model(quantity_file=5, record=True):
    if record==True:
        record_audio(quantity_file, c.RECORD_SECONDS, c.FILE_NAME_TRAIN, c.TRAIN_SET)

    print("Bắt đầu training")
    file_paths = open(c.FILE_NAME_TRAIN, 'r')
    features = np.asarray(())
    count = 1
    for path in file_paths:
        path = path.strip()
        print(path)

        sr, audio = read(c.TRAIN_SET + path)
        vector = ef(audio, sr)

        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == c.QUANTITY_TRAIN_FILE:
            gmm = GaussianMixture(n_components=6, max_iter=200, covariance_type='diag', n_init=10)
            gmm.fit(features)

            picklefile = path.split('-')[0] + '.gmm'
            pickle.dump(gmm, open(c.TRAINED_MODELS + picklefile, 'wb'))
            features = np.asarray(())
            count = 0
        count += 1


if __name__=="__main__":
    train_model(c.QUANTITY_TRAIN_FILE)
    print("trained successfull!")
    test_list_file(c.QUANTITY_TEST_FILE, True)
