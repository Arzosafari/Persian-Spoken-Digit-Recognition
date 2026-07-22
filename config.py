import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_PATH = os.path.join(BASE_DIR, "model_1d.h5")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder_1d.pkl")

SAMPLE_RATE = 16000
N_MFCC = 40
N_DELTA = 2
N_FEATURES = N_MFCC * (1 + N_DELTA)   
MAX_LEN = None         

BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42