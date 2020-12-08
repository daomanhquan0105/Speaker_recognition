from pyaudio import paInt16

FORMAT = paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 512
RECORD_SECONDS = 10
DEVICE_INDEX = 0

# dir
TRAIN_SET = 'training_set/'
TEST_SET = 'testing_set/'
TRAINED_MODELS = 'trained_models/'
QUANTITY_TRAIN_FILE = 60
QUANTITY_TEST_FILE = 15

# text
FILE_NAME_TRAIN = 'training_set_addition.txt'
FILE_NAME_TEST = 'testing_set_addition.txt'
