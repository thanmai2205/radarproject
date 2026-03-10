import streamlit as st
import numpy as np
import pandas as pd
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# ------------------------------
# Load Model
# ------------------------------

model = load_model("model.h5")

class_names = ["walking", "sitting", "standing", "fall"]

# ------------------------------
# Streamlit Page Config
# ------------------------------

st.set_page_config(page_title="Radar HAR System", layout="wide")

st.title("Radar-Based Human Activity Recognition System")

# ------------------------------
# Session State for Activity Log
# ------------------------------

if "activity_log" not in st.session_state:
    st.session_state.activity_log = {}

# ------------------------------
# Project Overview
# ------------------------------

st.header("Project Overview")

st.write("""
This system uses radar micro-Doppler spectrograms and deep learning models
to recognize human activities in indoor environments.

The system also detects abnormal events such as falls and provides
confidence-based predictions for intelligent indoor monitoring systems.
""")

# ------------------------------
# System Architecture
# ------------------------------

st.header("System Architecture")

st.write("""
Radar Signal → Spectrogram Generation → Deep Learning Model (CNN)
→ Activity Classification → Abnormal Activity Detection → Alert System
""")

# Optional architecture image
# st.image("architecture.png")

# ------------------------------
# Upload Spectrogram
# ------------------------------

st.header("Upload Radar Spectrogram")

uploaded_file = st.file_uploader(
    "Upload Spectrogram Image",
    type=["jpg", "jpeg", "png"]
)

# ------------------------------
# Prediction Section
# ------------------------------

if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Spectrogram", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Model prediction
    prediction = model.predict(img)

    label_index = np.argmax(prediction)
    confidence = float(np.max(prediction))

    label = class_names[label_index]

    st.subheader("Prediction Result")

    st.write("Predicted Activity:", label)

    st.write("Prediction Confidence:", round(confidence * 100, 2), "%")

    # ------------------------------
    # Abnormal Activity Detection
    # ------------------------------

    abnormal_classes = ["fall"]

    if label.lower() in abnormal_classes:

        st.error("⚠ Abnormal Activity Detected")

        st.write("Detection Time:", datetime.datetime.now())

    else:

        st.success("Normal Activity")

    # ------------------------------
    # Store Activity Log
    # ------------------------------

    if label in st.session_state.activity_log:
        st.session_state.activity_log[label] += 1
    else:
        st.session_state.activity_log[label] = 1


# ------------------------------
# Activity Analytics Dashboard
# ------------------------------

st.header("Activity Analytics")

if st.session_state.activity_log:

    df = pd.DataFrame(
        list(st.session_state.activity_log.items()),
        columns=["Activity", "Count"]
    )

    st.bar_chart(df.set_index("Activity"))

else:

    st.write("No activity data available yet.")

# ------------------------------
# Model Performance Section
# ------------------------------

st.header("Model Information")

st.write("Model Type: Convolutional Neural Network (CNN)")

st.write("Dataset Type: Radar Micro-Doppler Spectrogram")

st.write("Task: Human Activity Recognition")

# ------------------------------
# Download Activity Report
# ------------------------------

st.header("Download Activity Report")

if st.session_state.activity_log:

    report_df = pd.DataFrame(
        list(st.session_state.activity_log.items()),
        columns=["Activity", "Count"]
    )

    csv = report_df.to_csv(index=False)

    st.download_button(
        label="Download Activity Report",
        data=csv,
        file_name="activity_report.csv",
        mime="text/csv"
    )

# ------------------------------
# Live Monitoring Simulation
# ------------------------------

st.header("Live Monitoring Simulation")

if st.button("Start Monitoring"):
    st.write("System is monitoring activities in real-time...")

    st.info("Simulation Mode: Activities will be detected continuously.")

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")

st.write("Radar-Based Human Activity Recognition using Deep Learning")
