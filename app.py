import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64

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
    font-size:60px;
    font-weight:900;
    color:#1f2937;
}

.subtext {
    text-align:center;
    font-size:18px;
    color:#4b5563;
    margin-bottom:30px;
}

.card {
    padding:25px;
    border-radius:20px;
    background: rgba(255,255,255,0.75);
    backdrop-filter: blur(12px);
    box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
    transition: transform 0.3s ease;
}

.card:hover {
    transform: scale(1.02);
}

.alert-box {
    padding:20px;
    border-radius:15px;
    background-color:#ffccd5;
    color:#b00020;
    font-weight:700;
    text-align:center;
    animation: pulse 1.2s infinite;
}

@keyframes pulse {
    0% {box-shadow:0 0 0 0 rgba(255,0,0,0.4);}
    70% {box-shadow:0 0 0 20px rgba(255,0,0,0);}
    100% {box-shadow:0 0 0 0 rgba(255,0,0,0);}
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("radar_model.keras")
class_names = ['falling','sitting','walking']

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 Radar-Based Intelligent Surveillance Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtext">Real-time Radar Activity Recognition | AI Powered Fall Detection | Deep Learning Based Risk Analysis</div>',
    unsafe_allow_html=True
)

st.write("")

# ---------------- FILE UPLOADER ----------------
uploaded_file = st.file_uploader(
    "📡 Upload Radar Spectrogram Image",
    type=["png","jpg","jpeg"]
)

# ---------------- SESSION HISTORY ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PREDICTION ----------------
if uploaded_file:

    col1, col2 = st.columns([1,1])

    # -------- LEFT: IMAGE --------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        image = Image.open(uploaded_file).convert("RGB")  # FIXED RGB
        st.image(image, caption="Uploaded Spectrogram", use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- PREPROCESS --------
    img = image.resize((128,128))  # MATCH TRAINING SIZE
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -------- PREDICT --------
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # -------- RISK LOGIC --------
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

    # -------- RIGHT: RESULTS --------
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

        # -------- AUTO ALERT --------
        if risk == "HIGH" and confidence > 0.50:

            st.markdown(
                '<div class="alert-box">🚨 HIGH RISK ACTIVITY DETECTED!</div>',
                unsafe_allow_html=True
            )

            # Auto alarm sound (base64 method - cloud safe)
            with open("alarm.mp3", "rb") as f:
                audio_bytes = f.read()
                b64 = base64.b64encode(audio_bytes).decode()

                audio_html = f"""
                    <audio autoplay>
                        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                    </audio>
                """
                st.markdown(audio_html, unsafe_allow_html=True)

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
st.caption("AI Radar Surveillance System | Final Year Deep Learning Project | Deep Learning + Streamlit Cloud Deployment")

