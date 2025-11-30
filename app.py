import streamlit as st
from predict_utils import predict_image

st.title("**_ECG Heart Disease Detector_**")

uploaded_file = st.file_uploader("Upload ECG Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    prob = predict_image("temp.jpg")

    if prob > 0.5:
        st.success(f"Result: NORMAL ")
        st.markdown("Go enjoy a chocolate!!")
    else:
        st.error(f"Result: DISEASE DETECTED")
        st.markdown("Alert! Heart needs some TLC. Maybe less stress, more laughter")

