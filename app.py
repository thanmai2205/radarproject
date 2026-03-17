import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Radar Intelligent Surveillance", layout="wide")

# ---------------- LOGIN ----------------
def login():

    st.title("🔐 Radar Surveillance Login")

    users = {
        "admin":"radar123",
        "Thanmai":"tanu@123"
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
model = tf.keras.models.load_model("radar_model.keras")
class_names = ["falling","sitting","walking"]

# ---------------- SESSION STORAGE ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📡 Radar Control Panel")

menu = st.sidebar.radio(
    "Navigation",
    ["Dashboard","Upload Spectrogram","Detection History","System Info"]
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

    col1,col2,col3 = st.columns(3)

    col1.metric("Total Detections",len(st.session_state.history))
    col2.metric("Model","CNN")
    col3.metric("Classes","3")

    # ---------------- REAL AI RADAR ----------------
    import time
    import math

    st.subheader("📡 AI Powered Radar Simulation")

    radar_placeholder = st.empty()

    # Get last prediction
    if len(st.session_state.history) > 0:
        last = st.session_state.history[-1]["Activity"]
    else:
        last = None

    # Assign color + label
    if last == "falling":
        dot_color = "red"
        label = "FALLING"
    elif last == "sitting":
        dot_color = "yellow"
        label = "SITTING"
    elif last == "walking":
        dot_color = "lime"
        label = "WALKING"
    else:
        dot_color = "gray"
        label = "NO TARGET"

    target_x, target_y = 0.5, 0.3

    for angle in range(0, 360, 10):

        theta = math.radians(angle)

        fig = go.Figure()

        # Radar circle
        fig.add_shape(
            type="circle",
            x0=-1, y0=-1, x1=1, y1=1,
            line=dict(color="green", width=2)
        )

        # Sweep line
        fig.add_trace(go.Scatter(
            x=[0, math.cos(theta)],
            y=[0, math.sin(theta)],
            mode="lines",
            line=dict(color="lime", width=3)
        ))

        # Show target ONLY if prediction exists
        if last is not None:

            fig.add_trace(go.Scatter(
                x=[target_x],
                y=[target_y],
                mode="markers+text",
                marker=dict(color=dot_color, size=14),
                text=[label],
                textposition="top center",
                textfont=dict(color="white", size=12)
            ))

        fig.update_layout(
            showlegend=False,
            xaxis=dict(range=[-1.2,1.2], visible=False),
            yaxis=dict(range=[-1.2,1.2], visible=False),
            plot_bgcolor="black",
            paper_bgcolor="black",
            height=420
        )

        radar_placeholder.plotly_chart(fig, use_container_width=True)

        time.sleep(0.08)

# ---------------- UPLOAD PAGE ----------------
if menu == "Upload Spectrogram":

    uploaded_file = st.file_uploader(
        "Upload Radar Spectrogram",
        type=["png","jpg","jpeg"]
    )

    if uploaded_file:

        col1,col2 = st.columns(2)

        image = Image.open(uploaded_file)

        with col1:
            st.image(image,caption="Uploaded Spectrogram")

        # ---------- PREPROCESS ----------
        img = image.resize((160,160))
        img = np.array(img)/255.0
        img = np.expand_dims(img,axis=0)

        prediction = model.predict(img)

        scores = prediction[0]*100
        predicted_class = class_names[np.argmax(scores)]
        confidence = float(np.max(scores))

        # ---------- RISK ----------
        if predicted_class == "falling" and confidence > 75:
            risk = "HIGH"
            color = "red"
        elif predicted_class == "sitting":
            risk = "MEDIUM"
            color = "orange"
        else:
            risk = "LOW"
            color = "green"

        # ---------- SAVE HISTORY ----------
        st.session_state.history.append({
            "Activity":predicted_class,
            "Confidence":confidence
        })

        # ---------- RESULTS ----------
        with col2:

            st.subheader("Prediction Result")

            st.metric("Activity",predicted_class.upper())
            st.metric("Confidence",f"{confidence:.2f}%")
            st.metric("Risk Level",risk)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                gauge={'axis':{'range':[0,100]},
                       'bar':{'color':color}}
            ))

            st.plotly_chart(fig,use_container_width=True)

            # ---------- BUZZER ----------
            if predicted_class == "falling" and confidence > 75:

                st.error("🚨 HIGH RISK ACTIVITY DETECTED!")

                st.markdown("""
                <audio autoplay>
                <source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg">
                </audio>
                """, unsafe_allow_html=True)

        # ---------- ANALYTICS ----------
        st.subheader("Prediction Analytics")

        col3,col4 = st.columns(2)

        with col3:

            remaining = 100 - confidence

            pie_df = pd.DataFrame({
                "Label":[predicted_class,"Other Activities"],
                "Value":[confidence,remaining]
            })

            fig = px.pie(
                pie_df,
                names="Label",
                values="Value",
                title="Prediction Confidence Distribution",
                hole=0.4
            )

            st.plotly_chart(fig,use_container_width=True)

        with col4:

            score_df = pd.DataFrame({
                "Activity":class_names,
                "Confidence":scores
            })

            fig2 = px.bar(
                score_df,
                x="Activity",
                y="Confidence",
                color="Activity",
                title="Model Confidence for All Activities"
            )

            st.plotly_chart(fig2,use_container_width=True)

# ---------------- DETECTION HISTORY ----------------
if menu == "Detection History":

    st.subheader("Detection History")

    if len(st.session_state.history)==0:
        st.info("No detections yet")

    else:

        history_df = pd.DataFrame(st.session_state.history)

        st.dataframe(history_df)

        avg_conf = history_df.groupby("Activity")["Confidence"].mean().reset_index()

        fig = px.pie(
            avg_conf,
            names="Activity",
            values="Confidence",
            title="Average Confidence by Activity"
        )

        st.plotly_chart(fig,use_container_width=True)

        csv = history_df.to_csv(index=False)

        st.download_button(
            "Download Report",
            csv,
            "radar_detection_report.csv",
            "text/csv"
        )

# ---------------- SYSTEM INFO ----------------
if menu == "System Info":

    st.subheader("System Information")

    st.markdown("""
    **Project:** Radar Based Human Activity Recognition  
    **Model:** Convolutional Neural Network  
    **Activities:** Falling, Sitting, Walking  
    **Framework:** TensorFlow + Streamlit
    """)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("AI Radar Surveillance System ")
