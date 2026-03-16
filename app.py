import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------

st.set_page_config(page_title="Radar Surveillance AI", layout="wide")

# ---------------- LOGIN SYSTEM ----------------

def login():
st.title("🔐 Radar Surveillance System Login")

```
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):

    if username == "admin" and password == "radar123":
        st.session_state.logged_in = True
    else:
        st.error("Invalid credentials")
```

if "logged_in" not in st.session_state:
st.session_state.logged_in = False

if not st.session_state.logged_in:
login()
st.stop()

# ---------------- LOAD MODEL ----------------

model = tf.keras.models.load_model("radar_model.keras")
class_names = ['falling','sitting','walking']

# ---------------- SESSION STORAGE ----------------

if "history" not in st.session_state:
st.session_state.history = []

# ---------------- SIDEBAR ----------------

st.sidebar.title("📡 Radar Control Panel")

menu = st.sidebar.radio(
"Navigation",
["Dashboard","Upload Spectrogram","Detection History","System Info"]
)

st.sidebar.markdown("---")

if st.sidebar.button("Reset History"):
st.session_state.history = []

st.sidebar.markdown("### System Status")
st.sidebar.success("AI Model Active")

# ---------------- HEADER ----------------

st.title("🚀 Radar Based Intelligent Surveillance System")
st.caption("Deep Learning Radar Micro-Doppler Human Activity Recognition")

# ---------------- DASHBOARD ----------------

if menu == "Dashboard":

```
st.subheader("📊 System Overview")

col1,col2,col3 = st.columns(3)

col1.metric("Total Detections",len(st.session_state.history))
col2.metric("Model","CNN")
col3.metric("Classes","3")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    fig = px.histogram(
        history_df,
        x="Activity",
        color="Activity",
        title="Activity Detection Count"
    )

    st.plotly_chart(fig,use_container_width=True)
```

# ---------------- IMAGE UPLOAD ----------------

if menu == "Upload Spectrogram":

```
uploaded_file = st.file_uploader(
    "Upload Radar Spectrogram",
    type=["png","jpg","jpeg"]
)

if uploaded_file:

    col1,col2 = st.columns(2)

    image = Image.open(uploaded_file)

    with col1:
        st.image(image,caption="Uploaded Spectrogram")

    # preprocess
    img = image.resize((160,160))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))*100

    if predicted_class == "falling" and confidence > 75:
        risk="HIGH"
        color="red"
    elif predicted_class == "sitting":
        risk="MEDIUM"
        color="orange"
    else:
        risk="LOW"
        color="green"

    st.session_state.history.append({
        "Activity":predicted_class,
        "Confidence":confidence
    })

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

        if predicted_class=="falling" and confidence>75:
            st.error("🚨 HIGH RISK ACTIVITY DETECTED!")
```

# ---------------- HISTORY ----------------

if menu == "Detection History":

```
st.subheader("📜 Detection History")

if len(st.session_state.history)==0:
    st.info("No detections yet")

else:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(history_df)

    fig = px.pie(
        history_df,
        names="Activity",
        title="Activity Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

    csv = history_df.to_csv(index=False)

    st.download_button(
        "Download Report",
        csv,
        "radar_detection_report.csv",
        "text/csv"
    )
```

# ---------------- SYSTEM INFO ----------------

if menu == "System Info":

```
st.subheader("⚙ System Information")

st.markdown("""
**Project:** Radar Based Human Activity Recognition  
**Model:** Convolutional Neural Network  
**Classes:** Falling, Sitting, Walking  
**Framework:** TensorFlow + Streamlit  

This system detects human activities using radar micro-doppler spectrograms and deep learning.
""")
```
