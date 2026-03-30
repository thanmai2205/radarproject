import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import cv2
from scipy.signal import spectrogram

st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Radar Surveillance Login")

    users = {"admin": "radar123"}

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

# ---------------- LOAD MODEL ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "radar_model.keras")

model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)

model_classes = ["falling", "sitting", "walking"]

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Spectrogram", "Live Camera", "Detection History"]
)

# ---------------- HEADER ----------------
st.title("🚀 Radar Based Intelligent Surveillance")

# ---------------- SPECTROGRAM ----------------
def image_to_spectrogram(img1, img2):
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    diff = cv2.absdiff(gray1, gray2)
    diff = cv2.GaussianBlur(diff, (5,5), 0)

    motion = np.mean(diff)

    signal = (diff/255.0).flatten()
    f, t, Sxx = spectrogram(signal)

    Sxx = np.log(Sxx + 1e-10)
    Sxx = (Sxx - Sxx.min()) / (Sxx.max() - Sxx.min())

    spec_img = cv2.resize(Sxx, (160,160))
    spec_img = cv2.applyColorMap((spec_img*255).astype(np.uint8), cv2.COLORMAP_JET)

    spec_img = spec_img / 255.0
    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img, diff, motion

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.metric("Total Detections", len(st.session_state.history))

# ---------------- UPLOAD ----------------
elif menu == "Upload Spectrogram":

    st.subheader("Upload Two Images")

    img1 = st.file_uploader("Upload Frame 1", type=["png","jpg","jpeg"])
    img2 = st.file_uploader("Upload Frame 2", type=["png","jpg","jpeg"])

    if img1 and img2:
        image1 = Image.open(img1)
        image2 = Image.open(img2)

        st.image([image1, image2], caption=["Frame 1", "Frame 2"])

        spec_img, diff, motion = image_to_spectrogram(image1, image2)

        st.image(diff, caption="Motion Difference")
        st.image(spec_img[0], caption="Spectrogram")

        prediction = model.predict(spec_img)

        scores = prediction[0]*100
        base_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        # ---------------- CUSTOM LOGIC ----------------
        if motion < 2:
            predicted_class = "standing"

        elif base_class == "sitting":
            predicted_class = "standing"   # replace sitting

        else:
            predicted_class = base_class

        # ---------------- OUTPUT ----------------
        if predicted_class == "standing":
            st.info("🧍 Person is standing")

        elif predicted_class == "walking":
            st.success("🚶 Person is walking")

        elif predicted_class == "falling":
            st.error("🚨 EMERGENCY: PERSON IS FALLING!")
            st.audio("https://www.soundjay.com/buttons/beep-07.wav")

        st.write(f"Confidence: {confidence:.2f}%")

# ---------------- LIVE CAMERA ----------------
elif menu == "Live Camera":

    st.subheader("📷 Live AI Surveillance")

    img1 = st.camera_input("Frame 1")
    img2 = st.camera_input("Frame 2")

    if img1 and img2:
        image1 = Image.open(img1)
        image2 = Image.open(img2)

        spec_img, diff, motion = image_to_spectrogram(image1, image2)

        st.image(diff)
        st.image(spec_img[0])

        prediction = model.predict(spec_img)

        scores = prediction[0]*100
        base_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        # ---------------- FIX ----------------
        if motion < 2:
            predicted_class = "standing"

        elif base_class == "sitting":
            predicted_class = "standing"

        else:
            predicted_class = base_class

        # ---------------- DISPLAY ----------------
        if predicted_class == "standing":
            st.info("🧍 Standing")

        elif predicted_class == "walking":
            st.success("🚶 Walking")

        elif predicted_class == "falling":
            st.error("🚨 FALL DETECTED!")
            st.audio("https://www.soundjay.com/buttons/beep-07.wav")

        st.write(f"Confidence: {confidence:.2f}%")

        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

# ---------------- HISTORY ----------------
elif menu == "Detection History":
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
