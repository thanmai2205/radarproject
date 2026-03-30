import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import cv2
from scipy.signal import spectrogram
import plotly.graph_objects as go

# ---------------- PAGE ----------------
st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

# ----------- GLOBAL UI STYLE -----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main-title {
    font-size: 35px;
    font-weight: bold;
    color: #38bdf8;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    color: white;
    margin-bottom: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.5);
}
.highlight {
    border: 3px solid red;
    border-radius: 10px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
def login():
    st.markdown('<div class="main-title">🔐 Radar Surveillance Login</div>', unsafe_allow_html=True)

    users = {
        "admin": "radar123",
        "Thanmai": "tanu@123"
    }

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and password == users[username]:
            st.session_state.logged_in = True
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- MODEL ----------------
model = tf.keras.models.load_model("radar_model.keras", compile=False, safe_mode=False)

model_classes = ["falling", "sitting", "walking"]

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Live Camera", "Upload Spectrogram", "Detection History", "System Info"]
)

if st.sidebar.button("Reset History"):
    st.session_state.history = []

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 Radar Intelligent Surveillance System</div>', unsafe_allow_html=True)

# ---------------- SPECTROGRAM ----------------
def image_to_spectrogram(img1, img2):
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    motion_level = np.mean(diff)

    diff = diff / 255.0
    signal = diff.flatten()

    f, t, Sxx = spectrogram(signal, fs=100)
    Sxx = np.log(Sxx + 1e-10)

    Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))
    spec_img = cv2.resize(Sxx, (160,160))
    spec_img = cv2.applyColorMap((spec_img*255).astype(np.uint8), cv2.COLORMAP_JET)

    spec_img = spec_img / 255.0
    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img, diff, motion_level

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.metric("Total Detections", len(st.session_state.history))

    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(px.pie(df, names="Activity"))

        with col2:
            st.plotly_chart(px.bar(df, x="Activity", y="Confidence"))

# ---------------- LIVE CAMERA ----------------
elif menu == "Live Camera":

    st.subheader("📷 Live AI Surveillance")

    img1 = st.camera_input("Frame 1")
    img2 = st.camera_input("Frame 2")

    if img1 and img2:
        image1 = Image.open(img1)
        image2 = Image.open(img2)

        spec_img, diff, motion_level = image_to_spectrogram(image1, image2)

        st.image(diff)

        st.markdown('<div class="highlight">', unsafe_allow_html=True)
        st.image(spec_img[0])
        st.markdown('</div>', unsafe_allow_html=True)

        prediction = model.predict(spec_img)
        scores = prediction[0] * 100

        predicted_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        # Standing logic
        if motion_level < 2:
            predicted_class = "standing"

        if predicted_class == "sitting":
            predicted_class = "standing"

        st.success(f"Prediction: {predicted_class.upper()}")

        # 🚨 EMERGENCY SOUND
        if predicted_class == "falling" and confidence > 75:
            st.error("🚨 FALL DETECTED!")
            st.markdown("""
            <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg">
            </audio>
            """, unsafe_allow_html=True)

        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

# ---------------- UPLOAD ----------------
elif menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader("Upload Spectrogram")

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

        img = image.resize((160,160))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)
        scores = prediction[0]*100

        predicted_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        st.success(predicted_class.upper())

        # 🚨 EMERGENCY SOUND
        if predicted_class == "falling" and confidence > 75:
            st.error("🚨 FALL DETECTED!")
            st.markdown("""
            <audio autoplay>
            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg">
            </audio>
            """, unsafe_allow_html=True)

# ---------------- HISTORY ----------------
elif menu == "Detection History":
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

# ---------------- SYSTEM INFO ----------------
elif menu == "System Info":

    st.markdown('<div class="card">📡 AI Radar Surveillance System</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">🧠 CNN Model for Activity Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="card">⚙️ Features: Detection, Alerts, Analytics</div>', unsafe_allow_html=True)
