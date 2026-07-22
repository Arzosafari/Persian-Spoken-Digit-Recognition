import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder
import config
from config import *

PERSIAN_DIGITS = '۰۱۲۳۴۵۶۷۸۹'
PRE_EMPHASIS = 0.97
MIN_AUDIO_SAMPLES = int(0.05 * SAMPLE_RATE)

def persian_to_int(ch):
    if ch in PERSIAN_DIGITS:
        return PERSIAN_DIGITS.index(ch)
    return int(ch)

def apply_preemphasis(audio, coef=PRE_EMPHASIS):
    if len(audio) < 2:
        return audio
    return np.append(audio[0], audio[1:] - coef * audio[:-1])

def remove_silence(y, sr, top_db=25):
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return y
    start, end = intervals[0][0], intervals[-1][1]
    return y[start:end]

def preprocess_audio(audio, sr=SAMPLE_RATE):
    """Unified audio preprocessing for training and inference."""
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    audio = apply_preemphasis(audio)
    audio = remove_silence(audio, SAMPLE_RATE)
    if len(audio) < MIN_AUDIO_SAMPLES:
        return None
    peak = np.max(np.abs(audio))
    if peak < 1e-6:
        return None
    audio = audio / peak
    return audio

def load_audio(file_path):
    y, orig_sr = librosa.load(file_path, sr=None, mono=True)
    y = preprocess_audio(y, orig_sr)
    if y is None:
        raise ValueError(f"No usable speech in {file_path}")
    return y

def normalize_features(feats):
    """Per-utterance CMVN — helps generalize across speakers and microphones."""
    mean = np.mean(feats, axis=0, keepdims=True)
    std = np.std(feats, axis=0, keepdims=True) + 1e-9
    return (feats - mean) / std

def extract_features(audio):
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=512, hop_length=160, win_length=400,
    )
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
    features = np.vstack([mfcc, delta1, delta2])
    return normalize_features(features.T)

def pad_features(feat, target_len):
    if feat.shape[0] >= target_len:
        return feat[:target_len, :]
    pad = target_len - feat.shape[0]
    return np.pad(feat, ((0, pad), (0, 0)), mode='constant')

def audio_to_features(audio, sr=SAMPLE_RATE, target_len=None):
    """Full pipeline: raw audio -> padded feature matrix."""
    audio = preprocess_audio(audio, sr)
    if audio is None:
        return None
    feats = extract_features(audio)
    if target_len is not None:
        feats = pad_features(feats, target_len)
    return feats

def predict_digit(model, label_encoder, audio, sr=SAMPLE_RATE, use_tta=True):
    """
    Predict a spoken digit from raw audio.
    use_tta: test-time augmentation — average predictions over slight variants.
    """
    target_len = model.input_shape[1]
    processed = preprocess_audio(audio, sr)
    if processed is None:
        return None, 0.0, []

    variants = [processed]
    if use_tta:
        noise = processed + np.random.normal(0, 0.003, len(processed))
        variants.append(noise / (np.max(np.abs(noise)) + 1e-9))
        for rate in (0.92, 1.08):
            stretched = librosa.effects.time_stretch(y=processed, rate=rate)
            variants.append(stretched / (np.max(np.abs(stretched)) + 1e-9))
        shifted = np.roll(processed, int(0.01 * SAMPLE_RATE))
        variants.append(shifted / (np.max(np.abs(shifted)) + 1e-9))

    all_probs = []
    for variant in variants:
        feats = extract_features(variant)
        feats = pad_features(feats, target_len)
        feats = np.expand_dims(feats, axis=0)
        probs = model.predict(feats, verbose=0)[0]
        all_probs.append(probs)

    probs = np.mean(all_probs, axis=0)
    pred = int(np.argmax(probs))
    confidence = float(probs[pred])
    digit = label_encoder.inverse_transform([pred])[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    top3 = [(label_encoder.inverse_transform([i])[0], float(probs[i])) for i in top3_idx]
    return digit, confidence, top3

def scan_dataset(data_dir):
    audios, labels = [], []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in sorted(os.listdir(label_dir)):
            if fname.lower().endswith('.wav'):
                file_path = os.path.join(label_dir, fname)
                try:
                    audio = load_audio(file_path)
                    audios.append(audio)
                    labels.append(label)
                except ValueError:
                    print(f"Skipping empty file: {file_path}")
    return audios, labels

def prepare_dataset(data_dir):
    print("Loading audio files...")
    audios, raw_labels = scan_dataset(data_dir)
    labels_int = [persian_to_int(l) for l in raw_labels]
    le = LabelEncoder()
    encoded_labels = le.fit_transform(labels_int)
    num_classes = len(le.classes_)

    feat_lengths = []
    features_list = []
    print("Extracting features...")
    for audio in audios:
        feats = extract_features(audio)
        features_list.append(feats)
        feat_lengths.append(feats.shape[0])

    config.MAX_LEN = int(np.percentile(feat_lengths, 95))
    print(f"Padding length (95th percentile): {config.MAX_LEN}")

    X = np.zeros((len(features_list), config.MAX_LEN, N_FEATURES))
    for i, feats in enumerate(features_list):
        length = min(feats.shape[0], config.MAX_LEN)
        X[i, :length, :] = feats[:length, :]

    y = np.eye(num_classes)[encoded_labels]
    return X, y, le, audios
