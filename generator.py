import numpy as np
import librosa
import keras
import config
from data_utils import extract_features, pad_features

class DataGenerator(keras.utils.Sequence):
    def __init__(self, audio_list, labels, batch_size=32, shuffle=True, augment=False):
        super().__init__()
        self.audio_list = audio_list
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(audio_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.audio_list) / self.batch_size))

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, len(self.audio_list))
        batch_indices = self.indices[start:end]

        batch_X = []
        for i in batch_indices:
            audio = self.audio_list[i].copy()
            if self.augment:
                audio = self._augment_raw(audio)
            feats = extract_features(audio)
            if self.augment:
                feats = self._augment_features(feats)
            feats = pad_features(feats, config.MAX_LEN)
            batch_X.append(feats)
        batch_X = np.array(batch_X)
        batch_y = self.labels[batch_indices]
        return batch_X, batch_y

    def _augment_raw(self, audio):
        choice = np.random.choice(
            ['noise', 'stretch', 'pitch', 'volume', 'shift', 'none'],
            p=[0.25, 0.2, 0.2, 0.15, 0.1, 0.1],
        )
        if choice == 'noise':
            level = np.random.uniform(0.002, 0.01)
            audio = audio + np.random.normal(0, level, len(audio))
        elif choice == 'stretch':
            rate = np.random.uniform(0.85, 1.15)
            audio = librosa.effects.time_stretch(y=audio, rate=rate)
        elif choice == 'pitch':
            n_steps = np.random.randint(-4, 5)
            audio = librosa.effects.pitch_shift(
                y=audio, sr=config.SAMPLE_RATE, n_steps=n_steps,
            )
        elif choice == 'volume':
            audio = audio * np.random.uniform(0.6, 1.4)
        elif choice == 'shift':
            shift = np.random.randint(-int(0.05 * config.SAMPLE_RATE),
                                      int(0.05 * config.SAMPLE_RATE))
            audio = np.roll(audio, shift)
        audio = audio / (np.max(np.abs(audio)) + 1e-9)
        return audio

    def _augment_features(self, feats):
        """SpecAugment-style masking on MFCC frames."""
        feats = feats.copy()
        t, f = feats.shape
        if t > 4 and np.random.random() < 0.5:
            width = np.random.randint(1, max(2, t // 6))
            start = np.random.randint(0, t - width)
            feats[start:start + width, :] = 0
        if f > 4 and np.random.random() < 0.3:
            width = np.random.randint(1, max(2, f // 8))
            start = np.random.randint(0, f - width)
            feats[:, start:start + width] = 0
        return feats

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
