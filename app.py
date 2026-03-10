import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Spacer
import io
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

# ---------------- DARK MODE TOGGLE ----------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    bg_color = "#0f172a"
    text_color = "white"
else:
    bg_color = "linear-gradient(to right, #eef2ff, #f8fafc)"
    text_color = "#1e293b"

# ---------------- CUSTOM CSS ----------------
st.markdown(f"""
<style>

.stApp {{
    background: {bg_color};
}}

.main-title {{
    text-align:center;
    font-size:70px;
    font-weight:900;
    color:{text_color};
    animation: glow 2s infinite alternate;
}}

@keyframes glow {{
    from {{ text-shadow: 0 0 10px #6366f1; }}
    to {{ text-shadow: 0 0 25px #3b82f6; }}
}}

.feature-box {{
    background: white;
    padding:18px;
    border-radius:15px;
    text-align:center;
    font-weight:600;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    transition: transform 0.3s ease;
}}

.feature-box:hover {{
    transform: scale(1.05);
}}

.card {{
    padding:25px;
    border-radius:18px;
    background: rgba(255,255,255,0.8);
    backdrop-filter: blur(10px);
    box-shadow: 0px 6px 18px rgba(0,0,0,0.1);
}}

.alert-box {{
    padding:25px;
    border-radius:16px;
    background-color:#ffccd5;
    color:#b00020;
    font-weight:800;
    font-size:22px;
    text-align:center;
    animation: pulse 1s infinite;
}}

@keyframes pulse {{
    0% {{box-shadow:0 0 0 0 rgba(255,0,0,0.5);}}
    70% {{box-shadow:0 0 0 20px rgba(255,0,0,0);}}
    100% {{box-shadow:0 0 0 0 rgba(255,0,0,0);}}
}}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">🚀 Radar Intelligent Surveillance System</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ---------------- FEATURE BOXES ----------------
colf1, colf2, colf3, colf4 = st.columns(4)

with colf1:
    st.markdown('<div class="feature-box">📡 Activity Recognition</div>', unsafe_allow_html=True)

with colf2:
    st.markdown('<div class="feature-box">🚨 Smart Risk Detection</div>', unsafe_allow_html=True)

with colf3:
    st.markdown('<div class="feature-box">📊 Analytics Dashboard</div>', unsafe_allow_html=True)

with colf4:
    st.markdown('<div class="feature-box">☁ Cloud Deployment</div>', unsafe_allow_html=True)

st.write("---")

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("radar_model.keras")
class_names = ['falling','sitting','walking']

if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- REAL TIME MONITOR MODE ----------------
realtime_mode = st.sidebar.checkbox("🔴 Real-Time Monitoring Mode")

uploaded_file = st.file_uploader("Upload Radar Spectrogram", type=["png","jpg","jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

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

    st.metric("Predicted Activity", predicted_class.upper())
    st.metric("Confidence Score", f"{confidence*100:.2f}%")
    st.metric("Risk Level", risk)

    # Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence*100,
        gauge={'axis': {'range': [0,100]}, 'bar': {'color': color}},
        title={'text': "Confidence Meter"}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # ALERT
    if risk == "HIGH" and confidence > 0.5:
        st.markdown('<div class="alert-box">🚨 HIGH RISK ACTIVITY DETECTED!</div>', unsafe_allow_html=True)

# ---------------- ADMIN DASHBOARD ----------------
if st.sidebar.checkbox("🛠 Admin Analytics Dashboard"):

    st.header("Admin Activity Monitoring Panel")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history, columns=["Activity"])

        fig1 = px.pie(df, names="Activity", title="Activity Distribution")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.histogram(df, x="Activity", color="Activity", title="Activity Frequency")
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(df)

# ---------------- PDF REPORT ----------------
if st.button("📄 Download PDF Report"):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Radar Surveillance Report", styles["Title"]))
    elements.append(Spacer(1, 0.5 * inch))
    elements.append(Paragraph(f"Last Detected Activity: {predicted_class}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence*100:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))

    doc.build(elements)

    st.download_button(
        label="Click to Download",
        data=buffer.getvalue(),
        file_name="Radar_Report.pdf",
        mime="application/pdf"
    )

# ---------------- FOOTER ----------------
st.write("---")
st.caption("AI Radar Surveillance | Advanced Deep Learning Project")
