import streamlit as st
import numpy as np
import pickle
import librosa
import keras
import io
from config import MODEL_PATH, ENCODER_PATH
from data_utils import predict_digit

st.set_page_config(
    page_title="Persian Digit Recognizer",
    page_icon="🎙️",
    layout="centered"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Vazirmatn', sans-serif; }
    .big-digit { font-size: 120px; font-weight: bold; text-align: center; color: #FF4B4B; margin: 20px 0; }
    .confidence { font-size: 24px; text-align: center; color: #888; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_model()

def show_prediction(digit, confidence, top3):
    if digit is not None:
        st.markdown(f'<div class="big-digit">{digit}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="confidence">Confidence: {confidence:.1%}</div>', unsafe_allow_html=True)
        if confidence < 0.6:
            st.warning("Low confidence — try speaking louder and closer to the microphone.")
        st.markdown("---")
        st.markdown("#### Top 3 Predictions:")
        cols = st.columns(3)
        for idx, (d, conf) in enumerate(top3):
            with cols[idx]:
                st.metric(label=f"#{idx+1}", value=d, delta=f"{conf:.1%}")
    else:
        st.error("⚠️ No speech detected. Please speak clearly and try again.")

st.title("🎙️ Persian Digit Recognizer")
st.markdown("تشخیص اعداد فارسی (۰ تا ۹) با صدای شما")
st.markdown("---")

tab1, tab2 = st.tabs(["🎤 Live Recording", "📁 Upload File"])

with tab1:
    st.markdown("### Record your voice")
    st.markdown("Click the microphone, say one Persian digit (۰–۹), then stop.")
    st.caption("Tip: speak clearly for about 1 second, close to the mic.")

    audio_bytes = st.audio_input("🎤 Click to record")

    if audio_bytes is not None:
        st.audio(audio_bytes, format='audio/wav')
        audio_np, sr = librosa.load(io.BytesIO(audio_bytes.read()), sr=None, mono=True)
        digit, confidence, top3 = predict_digit(model, label_encoder, audio_np, sr, use_tta=True)
        show_prediction(digit, confidence, top3)

with tab2:
    st.markdown("### Upload an audio file")
    st.markdown("Supported formats: `.wav`, `.m4a`, `.mp3`")

    uploaded_file = st.file_uploader("Choose a file", type=['wav', 'm4a', 'mp3'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')
        audio_np, sr = librosa.load(uploaded_file, sr=None, mono=True)
        digit, confidence, top3 = predict_digit(model, label_encoder, audio_np, sr, use_tta=True)
        show_prediction(digit, confidence, top3)

st.sidebar.title("About")
st.sidebar.markdown("""
This app recognizes **Persian spoken digits** (۰–۹).

**Pipeline:**
- Pre-emphasis + silence removal
- 40 MFCCs + delta + delta-delta + CMVN
- 1D CNN with test-time augmentation

Works with live microphone or uploaded files.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using TensorFlow & Streamlit")
