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

st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

st.markdown("""
<style>
.card {
    padding: 15px;
    border-radius: 10px;
    background-color: #1f2937;
    color: white;
    margin-bottom: 10px;
}
.highlight-img img {
    border: 4px solid red;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

def login():
    st.title(" Radar Surveillance Login")

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

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "radar_model.keras")

model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

model_classes = ["falling", "sitting", "walking"]
display_classes = ["falling", "sitting", "standing", "walking"]

if "history" not in st.session_state:
    st.session_state.history = []

st.sidebar.title("📡 Radar Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Live Camera", "Upload Spectrogram", "Detection History", "System Info"]
)

if st.sidebar.button("Reset History"):
    st.session_state.history = []

st.title("Radar Based Intelligent Surveillance")

def image_to_spectrogram(img1, img2):
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (5,5), 0)

    motion_level = np.mean(diff)

    diff = diff / 255.0

    # ✅ UPDATED SIGNAL (better motion representation)
    signal = cv2.resize(diff, (64,64)).flatten()

    # ✅ Radar-like spectrogram
    f, t, Sxx = spectrogram(signal, fs=100, nperseg=128, noverlap=100)
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sxx = 10 * np.log10(Sxx + 1e-10)

    Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))

    spec_img = cv2.resize(Sxx, (160,160))
    spec_img = cv2.applyColorMap((spec_img*255).astype(np.uint8), cv2.COLORMAP_JET)

    spec_img = spec_img / 255.0
    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img, diff, motion_level

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.metric("Total Detections", len(st.session_state.history))

# ---------------- LIVE CAMERA ----------------
elif menu == "Live Camera":

    st.subheader(" Live AI Surveillance")
    st.info("Capture 2 frames (move slightly)")

    img1 = st.camera_input("Frame 1")
    img2 = st.camera_input("Frame 2")

    if img1 and img2:
        image1 = Image.open(img1)
        image2 = Image.open(img2)

        st.image([image1, image2])

        spec_img, diff, motion_level = image_to_spectrogram(image1, image2)

        st.image(diff, caption="Motion Difference")

        st.markdown('<div class="highlight-img">', unsafe_allow_html=True)

        # ✅ PROFESSIONAL LAYOUT
        colA, colB = st.columns([2,1])

        with colA:
            st.image(spec_img[0], caption="Time-Doppler Spectrogram", width=650)

        with colB:
            st.markdown("""
            ### 📡 Radar Analysis
            • Standing → low energy  
            • Walking → periodic pattern  
            • Falling → sudden burst  
            """)

        st.markdown('</div>', unsafe_allow_html=True)

        # ✅ FIXED MODEL INPUT
        spec_img_fixed = cv2.cvtColor((spec_img[0]*255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        spec_img_fixed = cv2.resize(spec_img_fixed, (160,160)) / 255.0
        spec_img_fixed = spec_img_fixed.reshape(1,160,160,1)

        prediction = model.predict(spec_img_fixed)
        scores = prediction[0] * 100

        base_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        # ✅ BETTER THRESHOLD
        if motion_level < 5:
            predicted_class = "standing"
        else:
            predicted_class = base_class
            if predicted_class == "sitting":
                predicted_class = "standing"

        st.markdown(f"## 🧠 Prediction: {predicted_class.upper()}")
        st.markdown(f"### 🎯 Confidence: {confidence:.2f}%")

        # ✅ IMPRESSIVE FEATURE
        st.progress(int(confidence))

        st.markdown("### 🟢 System Status: ACTIVE")

        if predicted_class == "falling" and confidence > 75:
            st.error(" FALL DETECTED!")

        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

# ---------------- UPLOAD ----------------
elif menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader("Upload Radar Spectrogram", type=["png","jpg","jpeg"])

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

        st.metric("Activity",predicted_class.upper())
        st.metric("Confidence",f"{confidence:.2f}%")

# ---------------- HISTORY ----------------
elif menu == "Detection History":
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

# ---------------- SYSTEM INFO ----------------
elif menu == "System Info":
    st.markdown("## System Information Dashboard")
    st.caption("Radar Intelligent Surveillance System")
