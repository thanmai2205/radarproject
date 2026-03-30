import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
import cv2
from scipy.signal import spectrogram

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

# ---------------- LOGIN ----------------
def login():
    st.title("🔐 Radar Surveillance Login")

    users = {
        "admin": "radar123",
        "Thanmai": "tanu@123"
    }

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and password == users[username]:
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid username or password")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD MODEL ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "radar_model.keras")

model = tf.keras.models.load_model(
    model_path,
    compile=False,
    safe_mode=False
)

class_names = ["falling", "sitting", "walking"]

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")
menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Spectrogram", "Live Camera", "Detection History", "System Info"]
)

if st.sidebar.button("Reset History"):
    st.session_state.history = []

# ---------------- HEADER ----------------
st.title("🚀 Radar Based Intelligent Surveillance")
st.caption("AI Powered Human Activity Recognition")

# ---------------- IMPROVED SPECTROGRAM ----------------
def image_to_spectrogram(img1, img2):
    gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)

    # Motion detection
    diff = cv2.absdiff(gray1, gray2)

    # Smooth
    diff = cv2.GaussianBlur(diff, (5,5), 0)

    # Normalize
    diff = diff / 255.0

    # Flatten
    signal = diff.flatten()

    # Spectrogram
    f, t, Sxx = spectrogram(signal, fs=100)

    # Log scaling
    Sxx = np.log(Sxx + 1e-10)

    # Normalize
    Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))

    # Resize
    spec_img = cv2.resize(Sxx, (160,160))

    # Color map (🔥 makes it look real)
    spec_img = cv2.applyColorMap((spec_img*255).astype(np.uint8), cv2.COLORMAP_JET)

    spec_img = spec_img / 255.0
    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img, diff

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(st.session_state.history))
    col2.metric("Model", "CNN")
    col3.metric("Classes", "3")

# ---------------- UPLOAD ----------------
elif menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image)

        st.warning("Upload TWO images for motion-based spectrogram")

# ---------------- LIVE CAMERA ----------------
elif menu == "Live Camera":

    st.subheader("📷 Live AI Surveillance (2-Frame Motion Mode)")

    st.info("Capture TWO frames while moving slightly")

    img1 = st.camera_input("📸 Capture Frame 1")
    img2 = st.camera_input("📸 Capture Frame 2")

    if img1 and img2:
        image1 = Image.open(img1)
        image2 = Image.open(img2)

        st.image([image1, image2], caption=["Frame 1", "Frame 2"])

        # Generate spectrogram
        spec_img, diff = image_to_spectrogram(image1, image2)

        st.image(diff, caption="Motion Difference")
        st.image(spec_img[0], caption="Radar-like Spectrogram 🔥")

        # Prediction
        prediction = model.predict(spec_img)

        scores = prediction[0] * 100
        predicted_class = class_names[np.argmax(scores)]
        confidence = float(np.max(scores))

        # Risk logic
        if predicted_class == "falling" and confidence > 75:
            risk = "HIGH"
        elif predicted_class == "sitting":
            risk = "MEDIUM"
        else:
            risk = "LOW"

        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")
        st.warning(f"Risk Level: {risk}")

        if risk == "HIGH":
            st.error("🚨 FALL DETECTED!")

        # Save history
        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

# ---------------- HISTORY ----------------
elif menu == "Detection History":

    st.subheader("Detection History")

    if len(st.session_state.history) == 0:
        st.info("No detections yet")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)

        fig = px.pie(df, names="Activity", values="Confidence")
        st.plotly_chart(fig)

# ---------------- INFO ----------------
elif menu == "System Info":

    st.subheader("System Information")

    st.markdown("""
    **Project:** Radar Based Human Activity Recognition  
    **Model:** CNN  
    **Technique:** Motion-based Spectrogram Simulation  
    **Pipeline:** Camera → Motion → Spectrogram → Prediction  
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Final Year AI Project 🚀")
