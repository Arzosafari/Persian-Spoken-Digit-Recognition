"""
Fine-tune the model with the user's own voice recordings (my_digit(0-9).wav).
Run once after recording your digits to personalize recognition.
"""
import os
import glob
import numpy as np
import pickle
import librosa
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import config
from config import MODEL_PATH, ENCODER_PATH, DATA_DIR, SAMPLE_RATE
from data_utils import prepare_dataset, preprocess_audio, extract_features, pad_features
from generator import DataGenerator

USER_PATTERN = "my_digit*.wav"
FINETUNE_EPOCHS = 40
FINETUNE_LR = 1e-4

def load_user_recordings():
    audios, labels = [], []
    for path in sorted(glob.glob(USER_PATTERN)):
        basename = os.path.basename(path)
        if basename == "my_digit.wav":
            continue
        try:
            idx = int(basename.replace("my_digit(", "").replace(").wav", ""))
        except ValueError:
            continue
        if 0 <= idx <= 9:
            y, sr = librosa.load(path, sr=None, mono=True)
            audio = preprocess_audio(y, sr)
            if audio is not None:
                audios.append(audio)
                labels.append(idx)
                print(f"  Loaded {path} -> digit {idx}")
    return audios, labels

def main():
    if not os.path.exists(MODEL_PATH):
        print("Train the base model first: python train.py")
        return

    print("Loading base dataset...")
    X, y, le, raw_audios = prepare_dataset(DATA_DIR)
    num_classes = y.shape[1]

    print("Loading user recordings...")
    user_audios, user_labels = load_user_recordings()
    if not user_audios:
        print(f"No user recordings found matching {USER_PATTERN}")
        return

    all_audios = raw_audios + user_audios * 30
    user_onehot = np.eye(num_classes)[user_labels]
    all_labels = np.vstack([y] + [user_onehot] * 30)

    print(f"Total samples: {len(all_audios)} (dataset={len(raw_audios)}, user={len(user_audios)})")

    model = keras.models.load_model(MODEL_PATH)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINETUNE_LR),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    gen = DataGenerator(all_audios, all_labels, batch_size=16, augment=True, shuffle=True)

    callbacks = [
        EarlyStopping(monitor='loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
    ]

    model.fit(gen, epochs=FINETUNE_EPOCHS, callbacks=callbacks, verbose=1)

    model.save(MODEL_PATH)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"Fine-tuned model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
