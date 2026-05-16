# ============================================================
# Streamlit App - Speech Emotion Recognition
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib
import os
import tempfile
import pandas as pd
from tensorflow.keras.models import load_model

# --- Page Config ---
st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️", layout="wide")

# --- Custom Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.result-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 16px; padding: 2rem; text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.result-emoji { font-size: 4rem; }
.result-label { font-size: 1.8rem; font-weight: 700; color: #4ECDC4;
    text-transform: uppercase; letter-spacing: 2px; margin-top: 0.5rem; }
.result-conf { font-size: 1.1rem; color: #aaa; margin-top: 0.3rem; }
</style>
""", unsafe_allow_html=True)

# --- Emotion Info ---
EMOJIS = {
    'neutral': '😐', 'calm': '😌', 'happy': '😊', 'sad': '😢',
    'angry': '😠', 'fearful': '😨', 'disgust': '🤢', 'surprised': '😲'
}
COLORS = {
    'neutral': '#95a5a6', 'calm': '#3498db', 'happy': '#f39c12', 'sad': '#2c3e50',
    'angry': '#e74c3c', 'fearful': '#9b59b6', 'disgust': '#27ae60', 'surprised': '#e67e22'
}

# --- MFCC extraction (same as notebook) ---
def extract_mfcc(file_path, sr=22050, duration=3, n_mfcc=40, max_len=130):
    signal, sr = librosa.load(file_path, sr=sr, duration=duration)
    if len(signal) < sr * duration:
        signal = np.pad(signal, (0, sr * duration - len(signal)), mode='constant')
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    if mfccs.shape[1] < max_len:
        mfccs = np.pad(mfccs, ((0, 0), (0, max_len - mfccs.shape[1])))
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs, signal, sr

# --- Load Model (cached) ---
@st.cache_resource
def load_artifacts():
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    model = load_model(os.path.join(model_dir, 'cnn_lstm_best.keras'))
    scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
    le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
    return model, scaler, le

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown("# 🎙️ Speech Emotion Recognition")
    st.markdown("Upload a speech audio file and AI will detect the emotion using a **CNN-LSTM** model.")
    st.markdown("---")

    try:
        model, scaler, le = load_artifacts()
    except Exception as e:
        st.error(f"Model not found! Run the notebook first to train.\n\nError: {e}")
        return

    uploaded = st.file_uploader("Upload an audio file (.wav)", type=['wav', 'mp3', 'ogg', 'flac'])

    if uploaded is None:
        st.info("Upload a .wav file to get started!")
        st.markdown("### Supported Emotions")
        cols = st.columns(4)
        for i, (emo, emoji) in enumerate(EMOJIS.items()):
            with cols[i % 4]:
                st.markdown(f"<div style='text-align:center;padding:0.8rem;'>"
                            f"<span style='font-size:2rem;'>{emoji}</span><br>"
                            f"<b>{emo.capitalize()}</b></div>", unsafe_allow_html=True)
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        tmp.write(uploaded.getvalue())
        tmp_path = tmp.name

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Audio Playback")
        st.audio(uploaded, format='audio/wav')

        st.markdown("### Waveform")
        mfccs, signal, sr = extract_mfcc(tmp_path)
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(signal, sr=sr, ax=ax, color='#667eea')
        ax.set_title('Waveform', fontweight='bold', color='white')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("### Prediction")

        with st.spinner("Analyzing..."):
            mfcc_flat = mfccs.reshape(1, -1)
            mfcc_scaled = scaler.transform(mfcc_flat)
            mfcc_input = mfcc_scaled.reshape(1, 40, 130)
            mfcc_input = np.transpose(mfcc_input, (0, 2, 1))

            prediction = model.predict(mfcc_input, verbose=0)[0]
            pred_idx = np.argmax(prediction)
            pred_emotion = le.inverse_transform([pred_idx])[0]
            confidence = prediction[pred_idx] * 100

        emoji = EMOJIS.get(pred_emotion, '')
        color = COLORS.get(pred_emotion, '#4ECDC4')
        st.markdown(f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label" style="color:{color};">{pred_emotion}</div>
            <div class="result-conf">Confidence: {confidence:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### All Probabilities")
        pred_df = pd.DataFrame({
            'Emotion': le.classes_, 'Confidence': prediction * 100
        }).sort_values('Confidence', ascending=True)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.barh(pred_df['Emotion'], pred_df['Confidence'],
                color=[COLORS.get(e, '#666') for e in pred_df['Emotion']])
        ax.set_xlabel('Confidence (%)', color='white')
        ax.set_facecolor('#0e1117')
        fig.patch.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        st.pyplot(fig)
        plt.close()

    os.unlink(tmp_path)

    st.markdown("---")
    st.markdown("<p style='text-align:center;color:#555;'>Built with TensorFlow, Librosa & Streamlit</p>",
                unsafe_allow_html=True)

if __name__ == '__main__':
    main()
