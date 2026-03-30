import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

# ---------------- FIX KERAS COMPATIBILITY ----------------
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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

# ---------------- SESSION STORAGE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")
st.sidebar.info("Click 'Live Camera' → Start Camera for detection")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Spectrogram", "Live Camera", "Detection History", "System Info"]
)

if st.sidebar.button("Reset History"):
    st.session_state.history = []

st.sidebar.success("AI Model Active")

# ---------------- HEADER ----------------
st.title("🚀 Radar Based Intelligent Surveillance")
st.caption("AI Powered Human Activity Recognition")

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":
    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Detections", len(st.session_state.history))
    col2.metric("Model", "CNN")
    col3.metric("Classes", "3")

# ---------------- UPLOAD PAGE ----------------
elif menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader("Upload Radar Spectrogram", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Uploaded Spectrogram")

        img = image.resize((160, 160))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

        scores = prediction[0] * 100
        predicted_class = class_names[np.argmax(scores)]
        confidence = float(np.max(scores))

        if predicted_class == "falling" and confidence > 75:
            risk = "HIGH"
            color = "red"
        elif predicted_class == "sitting":
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "green"

        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

        with col2:
            st.subheader("Prediction Result")

            st.metric("Activity", predicted_class.upper())
            st.metric("Confidence", f"{confidence:.2f}%")
            st.metric("Risk Level", risk)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': color}}
            ))

            st.plotly_chart(fig, use_container_width=True)

# ---------------- LIVE CAMERA (CLOUD) ----------------
elif menu == "Live Camera":

    st.subheader("📷 Live AI Surveillance (Cloud Mode)")

    img_file = st.camera_input("📸 Capture Image")

    if img_file is not None:
        image = Image.open(img_file)
        st.image(image, caption="Captured Image")

        img = image.resize((160,160))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)

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

# ---------------- DETECTION HISTORY ----------------
elif menu == "Detection History":

    st.subheader("Detection History")

    if len(st.session_state.history) == 0:
        st.info("No detections yet")
    else:
        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df)

        fig = px.pie(
            history_df,
            names="Activity",
            values="Confidence",
            title="Activity Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------- SYSTEM INFO ----------------
elif menu == "System Info":

    st.subheader("System Information")

    st.markdown("""
    **Project:** Radar Based Human Activity Recognition  
    **Model:** Convolutional Neural Network  
    **Activities:** Falling, Sitting, Walking  
    **Framework:** TensorFlow + Streamlit  
    **Mode:** Cloud Camera (Image Capture)
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI Radar Surveillance System | Final Year Project")
