# Speech Emotion Recognition (SER)

A deep learning project that recognizes **8 human emotions** from speech audio using **MFCC features** and a **CNN-LSTM hybrid** neural network.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.37-red?logo=streamlit)

---

## Detected Emotions

| Emotion | Emotion |
|---------|---------|
| Neutral | Angry |
| Calm | Fearful |
| Happy | Disgust |
| Sad | Surprised |

---

## Project Structure

```
speech_emotion_recognition/
├── data/
│   └── RAVDESS/              # Dataset (download separately)
│       ├── Actor_01/
│       ├── Actor_02/
│       └── ...
├── models/                    # Saved after training
│   ├── cnn_lstm_best.keras
│   ├── scaler.pkl
│   └── label_encoder.pkl
├── Speech_Emotion_Recognition.ipynb   # Main notebook (train & evaluate)
├── app.py                             # Streamlit web app (deploy)
├── requirements.txt
└── README.md
```

---

## Dataset

**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

- **Download:** [Kaggle - RAVDESS](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- **Size:** 1440 audio files (.wav)
- **Actors:** 24 (12 male, 12 female)
- **Emotions:** 8 (neutral, calm, happy, sad, angry, fearful, disgust, surprised)

**Setup:** Download and extract into `data/RAVDESS/` so that `Actor_01`, `Actor_02`, ... folders are directly inside.

---

## Model Architecture

We use a **CNN-LSTM Hybrid** - the best architecture for speech emotion recognition:

```
Input (130 timesteps x 40 MFCC features)
    |
    +-- Conv1D (64 filters) -> BatchNorm -> MaxPool -> Dropout
    +-- Conv1D (128 filters) -> BatchNorm -> MaxPool -> Dropout
    |
    +-- Bidirectional LSTM (64 units) -> Dropout
    |
    +-- Dense (128) -> BatchNorm -> Dropout
    +-- Dense (8, softmax) -> Output (emotion probabilities)
```

**Why CNN-LSTM?**
- **CNN layers** extract local spectral patterns from MFCCs
- **LSTM layers** capture temporal dependencies across time frames
- Combined, they outperform standalone CNN or LSTM models

---

## Features Extracted

| Feature | Dimensions | Description |
|---------|-----------|-------------|
| **MFCC** | 40 coefficients x 130 frames | Mel-Frequency Cepstral Coefficients - captures timbral texture of speech |

Audio is loaded at **22050 Hz**, fixed to **3 seconds**, and MFCCs are computed using `librosa`.

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download [RAVDESS from Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio) and extract into `data/RAVDESS/`.

### 4. Train the model
Open `Speech_Emotion_Recognition.ipynb` in Jupyter Notebook and **run all cells**.

### 5. Launch the web app
```bash
streamlit run app.py
```

---

## Notebook Steps

| Step | Description |
|------|-------------|
| **Step 1** | Import all libraries |
| **Step 2** | Load RAVDESS dataset and extract emotion labels from filenames |
| **Step 3** | EDA - emotion distribution, waveforms, and Mel spectrograms |
| **Step 4** | Extract MFCC features (40 coefficients x 130 time frames) |
| **Step 5** | Preprocess - encode labels, normalize, train/test split |
| **Step 6** | Build CNN-LSTM hybrid model |
| **Step 7** | Train with EarlyStopping and learning rate scheduling |
| **Step 8** | Evaluate - accuracy curves, confusion matrix, classification report |
| **Step 9** | Save model, scaler, and label encoder for deployment |

---

## Web App Features

- Upload .wav audio files
- Audio playback in browser
- Real-time waveform visualization
- Emotion prediction with confidence scores
- Probability bar chart for all 8 emotions

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| **Python 3.10+** | Programming language |
| **TensorFlow / Keras** | Deep learning framework |
| **Librosa** | Audio processing and MFCC extraction |
| **Scikit-learn** | Preprocessing and evaluation metrics |
| **Matplotlib / Seaborn** | Visualizations |
| **Streamlit** | Web app deployment |

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
librosa
soundfile
scikit-learn
tensorflow
streamlit
tqdm
joblib
```

---


