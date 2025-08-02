import os
import numpy as np
import pickle
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#MFCC Feature Extraction#
def extract_mfcc(file_path, n_mfcc=40, max_len=174):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc

#Extract label from RAVDESS filename#
def detect_emotion_from_filename(filename):
    
    emotion_code = filename.split("-")[2]
    emotion_map = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    return emotion_map.get(emotion_code, "unknown")

#Load Dataset#
def load_dataset(data_path):
    X, y = [], []
    for root, _, files in os.walk(data_path):
        for file in files:
            if file.endswith(".wav"):
                path = os.path.join(root, file)
                label = detect_emotion_from_filename(file)
                if label == "unknown":
                    continue
                print(f"Loading: {file} => {label}")
                mfcc = extract_mfcc(path)
                X.append(mfcc)
                y.append(label)
    return np.array(X), np.array(y)

#Train Model#
def train_model():
    data_path = "C:/Users/metti/OneDrive/Desktop/EmotionRecognitionApp/data"
    model_path = "C:/Users/metti/OneDrive/Desktop/EmotionRecognitionApp/models"
    os.makedirs(model_path, exist_ok=True)

    X, y = load_dataset(data_path)
    X = np.expand_dims(X, -1)  

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_onehot = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=X.shape[1:]),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(y_onehot.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))
    model.save(os.path.join(model_path, "emotion_cnn.keras"))
    with open(os.path.join(model_path, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)

    print("✅ Model and encoder saved in 'models/' folder")
#Predict Emotion#
def predict_emotion(audio_path):
    model_path = "C:/Users/metti/OneDrive/Desktop/EmotionRecognitionApp/models/emotion_cnn.keras"
    encoder_path = "C:/Users/metti/OneDrive/Desktop/EmotionRecognitionApp/models/label_encoder.pkl"

    model = load_model(model_path)
    with open(encoder_path, "rb") as f:
        le = pickle.load(f)

    mfcc = extract_mfcc(audio_path)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  
    pred = model.predict(mfcc)
    emotion = le.inverse_transform([np.argmax(pred)])[0]
    return emotion

#Main Interface#
if __name__ == "__main__":
    print("\n🎤 Emotion Recognition System")
    print("1. Train Model")
    print("2. Predict Emotion from Audio File")
    choice = input("Enter your choice (1/2): ")

    if choice == "1":
        train_model()

    elif choice == "2":
        file_path = input("Enter full path to .wav file: ").strip()
        if not os.path.exists(file_path):
            print("❌ Audio file not found.")
        elif not os.path.exists("C:/Users/metti/OneDrive/Desktop/EmotionRecognitionApp/models/emotion_cnn.keras"):
            print("❌ Model not found! Please train the model first.")
        else:
            emotion = predict_emotion(file_path)
            print(f"🎙️ Predicted Emotion: {emotion}")

    else:
        print("❌ Invalid choice.")
