# Emotion-Recognition
CodeAlpha Task 1
"Emotion Recognition from Speech using CNN"

This project is a deep learning-based Speech Emotion Recognition (SER) system that classifies human emotions such as happy, sad, angry, surprised, etc., from `.wav` audio files using MFCC features and a Convolutional Neural Network (CNN) model.


"Project Structure"

EmotionRecognitionApp/
│
├── data/ # Dataset folder (e.g., RAVDESS .wav files)
├── models/ # Trained model and label encoder saved here
├── emotion_recognition.py # Main Python file
├── README.md # Project description and usage instructions


"Features"

- Extracts **MFCC features** from `.wav` audio files
- Detects emotion labels based on **RAVDESS filename structure**
- Trains a **CNN model** to classify emotions
- Supports model **training and prediction** through command-line interface
- Saves trained model (`emotion_cnn.keras`) and label encoder (`label_encoder.pkl`) in `models/` directory

 "Emotions Supported"
   
 01   - Neutral   
 02   - Calm      
 03   - Happy     
 04   - Sad       
 05   - Angry     
 06   - Fearful   
 07   - Disgust   
 08   - Surprised 

"How to Use"

1. Clone the Repository

git clone https://github.com/yourusername/EmotionRecognitionApp.git
cd EmotionRecognitionApp

2. Install Required Libraries
Ensure Python 3.10 is installed. Then run:

pip install numpy librosa scikit-learn tensorflow

3. Prepare Dataset

Place '.wav' files in the data/ folder.
Filenames must follow the RAVDESS naming format like: 03-01-05-01-01-01-01.wav

4. Run the Script
> To Train the Model:
python emotion_recognition.py
# Enter: 1
Trained files will be saved in models/ directory.

> To Predict Emotion:
python emotion_recognition.py
# Enter: 2
# Provide full path to your .wav file
The predicted emotion will be shown in the terminal.

"Output"
models/emotion_cnn.keras — the trained CNN model
models/label_encoder.pkl — the saved label encoder
Console output with predicted emotion

"Model Architecture"
Conv2D (32 filters) → MaxPooling2D → Dropout
Conv2D (64 filters) → MaxPooling2D → Dropout
Flatten → Dense (128) → Dropout → Dense (output layer with softmax)

"Notes"
Model uses 40 MFCC features per file
Pads or trims features to a length of 174 frames
Audio must be mono-channel (librosa.load() handles this)

"License"
This project is licensed under the MIT License.
You are free to use, modify, and distribute with proper attribution.

"Credits"
Dataset: RAVDESS
Libraries: TensorFlow, Librosa, scikit-learn
Developed by: Your Name
