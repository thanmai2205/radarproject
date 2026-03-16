import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

# ---------------- LOGIN FUNCTION ----------------
def login():
    st.title("🔐 Radar Surveillance System Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "radar123":
            st.session_state.logged_in = True
            st.success("Login Successful")
        else:
            st.error("Invalid username or password")

# ---------------- LOGIN STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("radar_model.keras")
class_names = ['falling', 'sitting', 'walking']

# ---------------- SESSION STORAGE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Upload Spectrogram", "Detection History", "System Info"]
)

st.sidebar.markdown("---")

if st.sidebar.button("Reset History"):
    st.session_state.history = []

st.sidebar.success("AI Model Active")

# ---------------- HEADER ----------------
st.title("🚀 Radar-Based Intelligent Surveillance System")
st.caption("AI Powered Human Activity Recognition using Radar Micro-Doppler Spectrograms")

# ---------------- DASHBOARD ----------------
if menu == "Dashboard":

    st.subheader("📊 System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Detections", len(st.session_state.history))
    col2.metric("Model", "CNN")
    col3.metric("Classes", "3")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(st.session_state.history)

        fig = px.histogram(
            history_df,
            x="Activity",
            color="Activity",
            title="Activity Detection Count"
        )

        st.plotly_chart(fig, use_container_width=True)

# ---------------- UPLOAD PAGE ----------------
if menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader(
        "📡 Upload Radar Spectrogram",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:

        col1, col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.image(image, caption="Uploaded Spectrogram")

        # -------- PREPROCESS IMAGE --------
        img = image.resize((160,160))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))*100

        # -------- RISK LOGIC --------
        if predicted_class == "falling" and confidence > 75:
            risk = "HIGH"
            color = "red"
        elif predicted_class == "sitting":
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "green"

        # -------- STORE HISTORY --------
        st.session_state.history.append({
            "Activity": predicted_class,
            "Confidence": confidence
        })

        with col2:

            st.subheader("📊 Prediction Result")

            st.metric("Activity", predicted_class.upper())
            st.metric("Confidence", f"{confidence:.2f}%")
            st.metric("Risk Level", risk)

            # -------- CONFIDENCE GAUGE --------
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': color}
                },
                title={'text': "Confidence Meter"}
            ))

            st.plotly_chart(fig, use_container_width=True)

            if predicted_class == "falling" and confidence > 75:
                st.error("🚨 HIGH RISK ACTIVITY DETECTED!")

        # -------- ANALYTICS --------
        st.subheader("📈 Prediction Analytics")

        col3, col4 = st.columns(2)

        # PIE CHART
        with col3:

            remaining = 100 - confidence

            pie_df = pd.DataFrame({
                "Label": [predicted_class, "Other Activities"],
                "Value": [confidence, remaining]
            })

            fig = px.pie(
                pie_df,
                names="Label",
                values="Value",
                title="Prediction Confidence Distribution",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)

        # BAR CHART
        with col4:

            scores = prediction[0]*100

            score_df = pd.DataFrame({
                "Activity": class_names,
                "Confidence": scores
            })

            fig2 = px.bar(
                score_df,
                x="Activity",
                y="Confidence",
                color="Activity",
                title="Model Confidence for All Activities"
            )

            st.plotly_chart(fig2, use_container_width=True)

# ---------------- HISTORY PAGE ----------------
if menu == "Detection History":

    st.subheader("📜 Detection History")

    if len(st.session_state.history) == 0:
        st.info("No detections yet")

    else:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df)

        fig = px.pie(
            history_df,
            names="Activity",
            title="Activity Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

        csv = history_df.to_csv(index=False)

        st.download_button(
            "⬇ Download Report",
            csv,
            "radar_detection_report.csv",
            "text/csv"
        )

# ---------------- SYSTEM INFO ----------------
if menu == "System Info":

    st.subheader("⚙ System Information")

    st.markdown("""
    **Project:** Radar-Based Human Activity Recognition  

    **Model:** Convolutional Neural Network (CNN)  

    **Activities Detected:**
    - Falling
    - Sitting
    - Walking

    **Technologies Used:**
    - TensorFlow
    - Streamlit
    - Plotly
    - Python

    This system detects human activities using radar micro-doppler spectrogram images and deep learning.
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI Radar Surveillance System | Final Year Deep Learning Project")
