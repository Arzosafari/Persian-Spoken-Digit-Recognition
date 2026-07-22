import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import config
from config import *
from data_utils import prepare_dataset
from generator import DataGenerator
from model import build_model

def main():
    X, y, le, raw_audios = prepare_dataset(DATA_DIR)
    num_classes = y.shape[1]

    indices = np.arange(len(raw_audios))
    train_idx, val_idx = train_test_split(
        indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED,
        stratify=y.argmax(axis=1))

    train_audios = [raw_audios[i] for i in train_idx]
    val_audios = [raw_audios[i] for i in val_idx]
    y_train = y[train_idx]
    y_val = y[val_idx]

    print(f"Train: {len(train_audios)}, Val: {len(val_audios)}")
    print(f"Input shape: ({config.MAX_LEN}, {config.N_FEATURES})")

    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(num_classes),
        y=y_train.argmax(axis=1),
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print("Class weights:", class_weight_dict)

    model = build_model((config.MAX_LEN, config.N_FEATURES), num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    train_gen = DataGenerator(
        train_audios, y_train, batch_size=BATCH_SIZE, augment=True, shuffle=True,
    )
    val_gen = DataGenerator(
        val_audios, y_val, batch_size=BATCH_SIZE, augment=False, shuffle=False,
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1,
    )

    model.save(MODEL_PATH)
    with open(ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Encoder saved to {ENCODER_PATH}")

if __name__ == "__main__":
    main()
