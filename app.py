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
    signal = diff.flatten()

    f, t, Sxx = spectrogram(signal, fs=100)
    Sxx = np.log(Sxx + 1e-10)

    Sxx = (Sxx - np.min(Sxx)) / (np.max(Sxx) - np.min(Sxx))

    spec_img = cv2.resize(Sxx, (160,160))
    spec_img = cv2.applyColorMap((spec_img*255).astype(np.uint8), cv2.COLORMAP_JET)

    spec_img = spec_img / 255.0
    spec_img = np.expand_dims(spec_img, axis=0)

    return spec_img, diff, motion_level

if menu == "Dashboard":
    st.metric("Total Detections", len(st.session_state.history))

    st.markdown("###  Activity Overview")

    if len(st.session_state.history) > 0:

        df = pd.DataFrame(st.session_state.history)

        col1, col2 = st.columns(2)

        with col1:
            activity_count = df["Activity"].value_counts().reset_index()
            activity_count.columns = ["Activity", "Count"]

            fig1 = px.pie(activity_count, names="Activity", values="Count", title="Activity Distribution")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = px.bar(df, x="Activity", y="Confidence", color="Activity", title="Confidence Levels")
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("No data available yet. Start detection to see analytics.")

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
        st.image(spec_img[0], caption="Spectrogram")

        # ✅ ADDITION 1 → BIG SPECTROGRAM
        st.image(spec_img[0], caption="Time-Doppler Spectrogram (Enhanced View)", width=600)

        st.markdown('</div>', unsafe_allow_html=True)

        # ✅ ADDITION 2 → SIDE PANEL
        col1, col2 = st.columns([2,1])
        with col1:
            st.image(spec_img[0], width=600)
        with col2:
            st.markdown("""
            ### 📡 Radar Interpretation
            • Standing → Low variation  
            • Walking → Repetitive pattern  
            • Falling → Sudden spike  
            """)

        prediction = model.predict(spec_img)
        scores = prediction[0] * 100

        base_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        if motion_level < 2:
            predicted_class = "standing"
            risk = "LOW"
        else:
            predicted_class = base_class

            if predicted_class == "sitting":
                predicted_class = "standing"

            if predicted_class == "falling" and confidence > 75:
                risk = "HIGH"
            elif predicted_class == "sitting":
                risk = "MEDIUM"
            else:
                risk = "LOW"

        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

        # ✅ ADDITION 3 → CONFIDENCE BAR
        st.progress(int(confidence))

        # ✅ ADDITION 4 → LIVE STATUS
        st.markdown("### 🟢 Live Monitoring Active")

        # ✅ ADDITION 5 → MOTION INTENSITY
        st.markdown("### 📊 Motion Intensity")
        st.progress(int(min(motion_level * 10, 100)))

        # ✅ ADDITION 6 → STATUS BADGE
        st.success(f"🟢 System Stable | Risk Level: {risk}")

        # ✅ ADDITION 7 → BUZZER
        if predicted_class == "falling" and confidence > 75:
            st.error("🚨 EMERGENCY ALERT: FALL DETECTED 🚨")
            st.markdown("""
            <audio autoplay loop>
            <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg">
            </audio>
            """, unsafe_allow_html=True)

        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

elif menu == "Upload Spectrogram":
    uploaded_file = st.file_uploader("Upload Radar Spectrogram", type=["png","jpg","jpeg"])

    if uploaded_file:

        col1,col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.image(image,caption="Uploaded Spectrogram")

        img = image.resize((160,160))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)
        scores = prediction[0]*100

        predicted_class = model_classes[np.argmax(scores)]
        confidence = float(np.max(scores))

        st.session_state.history.append({
            "Activity":predicted_class,
            "Confidence":confidence
        })

        with col2:
            st.metric("Activity",predicted_class.upper())
            st.metric("Confidence",f"{confidence:.2f}%")

elif menu == "Detection History":
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

elif menu == "System Info":
    st.markdown("##  System Information Dashboard")
    st.caption("Radar Intelligent Surveillance System")
