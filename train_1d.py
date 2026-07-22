import os
import numpy as np
import pickle
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras
from keras import layers


DATA_DIR = "dataset"
SAMPLE_RATE = 16000
N_MFCC = 40
N_DELTA = 2
N_FEATURES = N_MFCC * (1 + N_DELTA)   # 120
MAX_LEN = None                        
BATCH_SIZE = 32
EPOCHS = 80
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

MODEL_FILE = "model_1d.h5"
ENCODER_FILE = "label_encoder_1d.pkl"


def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y

def extract_features(audio):
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC)  # (40, t)
    n_frames = mfcc.shape[1]
    if n_frames < 3:
        pad_width = 3 - n_frames
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='mean')
        n_frames = 3
    width = min(9, n_frames - 2)
    if width % 2 == 0:
        width -= 1
    delta1 = librosa.feature.delta(mfcc, order=1, width=width)
    delta2 = librosa.feature.delta(mfcc, order=2, width=width)
    features = np.vstack([mfcc, delta1, delta2])   # (120, t)
    return features.T   # (t, 120)

def pad_features(feat, target_len):
    if feat.shape[0] >= target_len:
        return feat[:target_len, :]
    else:
        pad = target_len - feat.shape[0]
        return np.pad(feat, ((0, pad), (0, 0)), mode='constant')

def prepare_dataset(data_dir):
    X, y = [], []
    lengths = []
    print("Loading audio files...")
    for digit in range(10):
        folder = os.path.join(data_dir, str(digit))
        for fname in os.listdir(folder):
            if fname.endswith('.wav'):
                audio = load_audio(os.path.join(folder, fname))
                feat = extract_features(audio)
                lengths.append(feat.shape[0])
                X.append(feat)
                y.append(digit)
    
    max_len = int(np.percentile(lengths, 95))
    print(f"Padding length: {max_len}")
    X_pad = np.array([pad_features(f, max_len) for f in X])
    y = np.array(y)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X_pad, tf.keras.utils.to_categorical(y_enc, 10), le, max_len


def augment_audio(audio, sr):
    """Randomly apply noise, time stretch, or pitch shift."""
    choice = np.random.choice(['none', 'noise', 'stretch', 'pitch'], p=[0.3, 0.3, 0.2, 0.2])
    if choice == 'noise':
        noise = np.random.normal(0, 0.005, len(audio))
        audio = audio + noise
    elif choice == 'stretch':
        rate = np.random.uniform(0.8, 1.2)
        audio = librosa.effects.time_stretch(y=audio, rate=rate)
    elif choice == 'pitch':
        n_steps = np.random.randint(-3, 4)
        audio = librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)
    return audio

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size=32, shuffle=True, augment=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_X = []
        batch_y = self.y[batch_idx]
        for i in batch_idx:
            
            pass