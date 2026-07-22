import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import keras
import config
from config import *
from data_utils import prepare_dataset, extract_features, pad_features

def main():
   
    X, y, le, raw_audios = prepare_dataset(DATA_DIR)
    num_classes = y.shape[1]

    
    indices = np.arange(len(raw_audios))
    _, val_idx = train_test_split(
        indices, test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED,
        stratify=y.argmax(axis=1))

    val_audios = [raw_audios[i] for i in val_idx]
    y_val = y[val_idx]

   
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded.")

   
    X_val = []
    for audio in val_audios:
        feats = extract_features(audio)
        feats = pad_features(feats, model.input_shape[1])  # pad to model's input length
        X_val.append(feats)
    X_val = np.array(X_val)

   
    probs = model.predict(X_val, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🔹 Accuracy: {acc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[str(i) for i in le.classes_]))

   
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    main()