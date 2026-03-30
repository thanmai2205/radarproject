import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

# ---------------- SPECTROGRAM FUNCTION ----------------
def image_to_spectrogram(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)

    signal = gray.flatten()

    f, t, Sxx = spectrogram(signal)

    spec_img = np.log(Sxx + 1)

    spec_img = (spec_img - spec_img.min()) / (spec_img.max() - spec_img.min())

    spec_img = cv2.resize(spec_img, (160, 160))

    spec_img = np.stack([spec_img]*3, axis=-1)

    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Detections", len(st.session_state.history))
    col2.metric("Model", "CNN")
    col3.metric("Classes", "3")

# ---------------- UPLOAD ----------------
elif menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image")

        spec_img = image_to_spectrogram(image)
        st.image(spec_img[0], caption="Generated Spectrogram")

        prediction = model.predict(spec_img)

        scores = prediction[0] * 100
        predicted_class = class_names[np.argmax(scores)]
        confidence = float(np.max(scores))

        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

# ---------------- LIVE CAMERA ----------------
elif menu == "Live Camera":

    st.subheader("📷 Live AI Surveillance (Cloud Mode)")

    img_file = st.camera_input("📸 Capture Image")

    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Captured Image")

        spec_img = image_to_spectrogram(image)
        st.image(spec_img[0], caption="Generated Spectrogram")

        prediction = model.predict(spec_img)

        scores = prediction[0] * 100
        predicted_class = class_names[np.argmax(scores)]
        confidence = float(np.max(scores))

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
    **Activities:** Falling, Sitting, Walking  
    **Mode:** Camera → Spectrogram → Prediction  
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Final Year AI Project 🚀")
