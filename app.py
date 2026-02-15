import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Radar Intelligent Surveillance",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(to right, #eef2ff, #f8fafc);
}

.main-title {
    text-align:center;
    font-size:48px;
    font-weight:800;
    color:#2b2d42;
    animation: fadeIn 1.5s ease-in;
}

@keyframes fadeIn {
    from {opacity:0;}
    to {opacity:1;}
}

.card {
    padding:25px;
    border-radius:18px;
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: scale(1.02);
}

.alert {
    padding:20px;
    border-radius:14px;
    background-color:#ffccd5;
    color:#b00020;
    font-weight:700;
    text-align:center;
    animation: pulse 1.2s infinite;
}

@keyframes pulse {
    0% {box-shadow:0 0 0 0 rgba(255,0,0,0.4);}
    70% {box-shadow:0 0 0 15px rgba(255,0,0,0);}
    100% {box-shadow:0 0 0 0 rgba(255,0,0,0);}
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
st.markdown('<p class="main-title">🚀 Radar-Based Intelligent Surveillance Dashboard</p>', unsafe_allow_html=True)
st.write("")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "📡 Upload Radar Spectrogram",
    type=["png","jpg","jpeg"]
)

# ---------------- PROCESS IMAGE ----------------
if uploaded_file:

    col1, col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Spectrogram", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    img = image.resize((160,160))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Risk logic
    if predicted_class == "falling":
        risk = "HIGH"
        color = "red"
    elif predicted_class == "sitting":
        risk = "MEDIUM"
        color = "orange"
    else:
        risk = "LOW"
        color = "green"

    st.session_state.history.append(predicted_class)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("📊 Prediction Summary")

        st.metric("Predicted Activity", predicted_class.upper())
        st.metric("Confidence Score", f"{confidence*100:.2f}%")
        st.metric("Risk Level", risk)

        # -------- CONFIDENCE GAUGE --------
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence*100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
            },
            title={'text': "Confidence Meter"}
        ))

        st.plotly_chart(fig, use_container_width=True)

        # -------- ALERT + WORKING ALARM --------
        if predicted_class == "falling" and confidence > 0.75:

            st.markdown(
                '<div class="alert">🚨 HIGH RISK ACTIVITY DETECTED!</div>',
                unsafe_allow_html=True
            )

            # Download alarm sound safely
            alarm_url = "https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg"
            response = requests.get(alarm_url)

            st.audio(BytesIO(response.content), format="audio/ogg")

        st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ANALYTICS ----------------
if st.session_state.history:

    st.write("")
    st.header("📈 Intelligent Surveillance Analytics")

    history_df = pd.DataFrame(st.session_state.history, columns=["Activity"])

    col3, col4 = st.columns(2)

    with col3:
        fig = px.pie(
            history_df,
            names="Activity",
            title="Activity Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig2 = px.histogram(
            history_df,
            x="Activity",
            color="Activity",
            title="Detected Activity Count"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- FOOTER ----------------
st.write("---")
st.caption("AI Radar Surveillance System | Final Year Deep Learning Project")
