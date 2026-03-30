```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import time
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

# ---------------- LOAD MODEL (SAFE) ----------------
model = None

try:
    model = tf.keras.models.load_model("radar_model.h5", compile=False)
except:
    st.error("❌ Model file not found. Please place radar_model.h5 in project folder")
    st.stop()

class_names = ["falling", "sitting", "walking"]

# ---------------- SESSION STORAGE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")
st.sidebar.info("Click 'Live Camera' → Start Camera for real-time detection")

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

# ---------------- SPECTROGRAM FUNCTION ----------------
def frame_to_spectrogram(frame, prev_frame=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        return None, gray

    diff = cv2.absdiff(prev_frame, gray)

    # Fake spectrogram (simulation)
    spec_img = cv2.merge([diff, diff, diff])
    spec_img = spec_img / 255.0
    spec_img = cv2.resize(spec_img, (160, 160))
    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img, gray

# ---------------- UPLOAD PAGE ----------------
if menu == "Upload Spectrogram":

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

# ---------------- LIVE CAMERA ----------------
if menu == "Live Camera":

    st.subheader("📷 Live AI Surveillance (Simulation Mode)")

    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])
    status = st.empty()

    cap = cv2.VideoCapture(0)
    prev_frame = None

    while run:
        ret, frame = cap.read()

        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.resize(frame, (320, 240))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        spec_img, prev_frame = frame_to_spectrogram(frame, prev_frame)

        if spec_img is None:
            continue

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

        # Display text
        status.markdown(f"""
        **Prediction:** {predicted_class.upper()}  
        **Confidence:** {confidence:.2f}%  
        **Risk Level:** {risk}
        """)

        # Progress bar
        st.progress(int(confidence))

        # Show label on frame
        cv2.putText(frame, predicted_class, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if predicted_class == "falling" and confidence > 75:
            st.error("🚨 FALL DETECTED!")

        time.sleep(0.2)

    cap.release()

# ---------------- DETECTION HISTORY ----------------
if menu == "Detection History":

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
if menu == "System Info":

    st.subheader("System Information")

    st.markdown("""
    **Project:** Radar Based Human Activity Recognition  
    **Model:** Convolutional Neural Network  
    **Activities:** Falling, Sitting, Walking  
    **Framework:** TensorFlow + Streamlit  
    **Mode:** Simulation (Webcam-based Spectrogram)
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI Radar Surveillance System | Final Year Project")
```
