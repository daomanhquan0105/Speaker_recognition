import os
import pickle
from scipy.io.wavfile import read
from Caculate import extract_features as ef
import numpy as np
import CONFIG as c
import time
import pandas as pd
from recordAudio import record_audio, record_one_file


def test_list_file(quantity_file=2, record=False):
    if record == True:
        record_audio(quantity_file, c.RECORD_SECONDS, c.FILE_NAME_TEST, c.TEST_SET, True)

    print("Bắt đầu kiểm tra")
    # gmm_files, models, speakers = listSpeaker()
    gmm_files = [os.path.join(c.TRAINED_MODELS, file_name) for file_name in os.listdir(c.TRAINED_MODELS)
                 if file_name.endswith('.gmm')]

    models = [pickle.load(open(file_name, 'rb')) for file_name in gmm_files]
    speakers = [file_name.split('\\')[-1].split(".gmm")[0] for file_name in gmm_files]
    file_paths = open(c.FILE_NAME_TEST, 'r')
    cols = ['file_name', 'scores', 'speaker']
    df_result = pd.DataFrame(columns=cols)
    # test_list_file(file_paths, models, speakers)
    for path in file_paths:
        path = path.strip()
        sr, audio = read(c.TEST_SET + path)
        vector = ef(audio, sr)

        log_likelihood = np.zeros(len(models))
        for i in range(len(models)):
            gmm = models[i]
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()

        winner = np.argmax(log_likelihood)
        speaker = speakers[winner].split('/')[-1]
        # print(f"{path} la {speaker} nói")
        new_data = [path, scores, speaker]
        new_df = pd.DataFrame([new_data], columns=cols)
        df_result = pd.concat([df_result, new_df], ignore_index=True)
    df_result.to_csv('res/result.csv')
    print("Sumit successfully!result to save dir res/result.csv")


def test_one_file(file_name="testfile.wav"):
    file_path = record_one_file(file_name)
    print("bắt đầu kiểm tra")
    gmm_files = [os.path.join(c.TRAINED_MODELS, file_name) for file_name in os.listdir(c.TRAINED_MODELS)
                 if file_name.endswith('.gmm')]

    models = [pickle.load(open(file_name, 'rb')) for file_name in gmm_files]
    speakers = [file_name.split('\\')[-1].split(".gmm")[0] for file_name in gmm_files]

    file_path = file_path.strip()
    sr, audio = read(file_path)
    vector = ef(audio, sr)

    log_likelihood = np.zeros(len(models))

    for i in range(len(models)):
        gmm = models[i]
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)

    speaker = speakers[winner].split('/')[-1]
    time.sleep(1)
    print(f"{file_name} la {speaker} nói")
