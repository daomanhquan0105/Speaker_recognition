import python_speech_features as mfcc
import numpy as np
from sklearn import preprocessing


def calculate_delta(array):
    """
    :param array: array input
    :return: array
    """
    rows, cols = array.shape
    deltas = np.zeros((rows, cols))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10
    return deltas


def extract_features(audio, rate):
    """
    :param audio: file wav input
    :param rate: learning rate: là một siêu tham số kiểm soát mức độ thay đổi mô hình để đáp ứng với lỗi ước tính mỗi khi trọng số mô hình được cập nhật
    :return: array combine : là kết hợ 2 mảng với nhau
    """
    mfcc_feature = mfcc.mfcc(audio, rate, 0.025, 0.01, 20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined
