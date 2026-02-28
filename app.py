import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Radar Intelligent Surveillance",
    layout="wide"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "ℹ️ About Us", "🔐 Login"]
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("radar_model.keras")

model = load_model()
class_names = ['falling','sitting','walking']

if "history" not in st.session_state:
    st.session_state.history = []

# =========================================================
# 🏠 HOME PAGE
# =========================================================
if page == "🏠 Home":

    st.markdown("""
    <h1 style='text-align:center;
    font-size:60px;
    font-weight:900;
    color:#1e293b;'>
    🚀 Radar-Based Intelligent Surveillance Dashboard
    </h1>
    """, unsafe_allow_html=True)

    st.write("---")

    uploaded_file = st.file_uploader(
        "📡 Upload Radar Spectrogram",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:

        col1, col2 = st.columns([1,1])

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Spectrogram", use_container_width=True)

        img = image.resize((160,160))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        # -------- Risk Logic --------
        if predicted_class == "falling" and confidence > 0.75:
            risk = "HIGH"
            color = "red"
        elif predicted_class == "falling" and confidence > 0.50:
            risk = "MEDIUM (Uncertain)"
            color = "orange"
        elif confidence < 0.50:
            risk = "LOW CONFIDENCE"
            color = "gray"
        elif predicted_class == "sitting":
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "green"

        st.session_state.history.append(predicted_class)

        with col2:
            st.subheader("📊 Prediction Summary")
            st.metric("Predicted Activity", predicted_class.upper())
            st.metric("Confidence Score", f"{confidence*100:.2f}%")
            st.metric("Risk Level", risk)

            st.subheader("🔎 Class Probabilities")
            for i, class_name in enumerate(class_names):
                st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")

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

            if risk == "HIGH":
                st.error("🚨 HIGH RISK ACTIVITY DETECTED!")

    # -------- Analytics --------
    if st.session_state.history:
        st.write("---")
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

# =========================================================
# ℹ️ ABOUT US PAGE
# =========================================================
elif page == "ℹ️ About Us":

    st.title("ℹ️ About This Project")

    st.write("""
    ### 📌 Project Title:
    Radar-Based Human Activity Recognition for Intelligent Surveillance Systems

    ### 🎯 Objective:
    To develop a deep learning-based radar surveillance system capable of:
    - Detecting human activities
    - Identifying abnormal events like falls
    - Providing confidence-based risk alerts

    ### 🧠 Technologies Used:
    - TensorFlow & Keras
    - CNN Architecture
    - Streamlit Deployment
    - Plotly Visualization

    ### 🚀 Applications:
    - Elderly fall detection
    - Defense monitoring
    - Smart home security
    - Healthcare monitoring

    This system ensures privacy by using radar micro-Doppler signals instead of cameras.
    """)

# =========================================================
# 🔐 LOGIN PAGE
# =========================================================
elif page == "🔐 Login":

    st.title("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "radar123":
            st.success("✅ Login Successful")
            st.info("Welcome to Admin Dashboard")
        else:
            st.error("❌ Invalid Credentials")

# ---------------- FOOTER ----------------
st.write("---")
st.caption("AI Radar Surveillance System | Final Year Deep Learning Project")
