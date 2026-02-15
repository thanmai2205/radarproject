import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from io import BytesIO

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Radar Intelligent Surveillance",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

/* Main Title */
.main-title {
    text-align:center;
    font-size:60px;
    font-weight:900;
    color:#1f2937;
}

/* Subtitle */
.sub-title {
    text-align:center;
    font-size:22px;
    color:#4b5563;
    margin-bottom:20px;
}

/* Feature Cards */
.feature-card {
    background: white;
    padding:20px;
    border-radius:15px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    text-align:center;
    font-weight:600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("radar_model.keras")
class_names = ['falling','sitting','walking']

# ---------------- SESSION STORAGE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 Radar-Based Intelligent Surveillance System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">AI-Powered Human Activity Detection using Radar Spectrogram Analysis</div>', unsafe_allow_html=True)

st.write("")

# ---------------- FEATURE SECTION ----------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="feature-card">📡 Real-time Activity Detection</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="feature-card">🧠 Deep Learning CNN Model</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="feature-card">🚨 Automatic Risk Alert System</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="feature-card">📊 Intelligent Analytics Dashboard</div>', unsafe_allow_html=True)

st.write("")
st.write("---")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📡 Upload Radar Spectrogram Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file:

    col_img, col_result = st.columns([1,1])

    with col_img:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Spectrogram", use_container_width=True)

    # 🔥 IMPORTANT: MODEL TRAINED ON 128x128
    img = image.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # ---------------- SMART RISK ENGINE ----------------
    if predicted_class == "falling" and confidence > 0.70:
        risk = "🔴 HIGH"
        alarm_trigger = True
    elif predicted_class == "falling":
        risk = "🟠 MEDIUM"
        alarm_trigger = False
    elif predicted_class == "sitting":
        risk = "🟡 LOW"
        alarm_trigger = False
    else:
        risk = "🟢 SAFE"
        alarm_trigger = False

    st.session_state.history.append(predicted_class)

    with col_result:
        st.subheader("📊 Prediction Summary")
        st.metric("Predicted Activity", predicted_class.upper())
        st.metric("Confidence Score", f"{confidence*100:.2f}%")
        st.metric("Risk Level", risk)

        # -------- CONFIDENCE GAUGE --------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence*100,
            gauge={'axis': {'range': [0, 100]}},
            title={'text': "Confidence Meter"}
        ))
        st.plotly_chart(fig, use_container_width=True)

        # -------- ALERT + AUTO SOUND --------
        if alarm_trigger:
            st.error("🚨 HIGH RISK ACTIVITY DETECTED!")

            alarm_url = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
            response = requests.get(alarm_url)

            st.audio(BytesIO(response.content), format="audio/ogg")

# ---------------- ANALYTICS ----------------
if st.session_state.history:

    st.write("")
    st.header("📈 Surveillance Analytics")

    history_df = pd.DataFrame(st.session_state.history, columns=["Activity"])

    col_a, col_b = st.columns(2)

    with col_a:
        fig1 = px.pie(history_df, names="Activity", title="Activity Distribution")
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.histogram(history_df, x="Activity", color="Activity", title="Activity Count")
        st.plotly_chart(fig2, use_container_width=True)

st.write("---")
st.caption("Final Year Deep Learning Project | Radar HAR Surveillance System")

