# import streamlit as st
# import tensorflow as tf
# from predict_utils import preprocess_image
# import os

# # 1. Page Configuration
# st.set_page_config(page_title="ECG Heart Disease Detector", page_icon="❤️")

# # 2. Load Model
# @st.cache_resource
# def load_my_model():
#     if not os.path.exists('heart_disease_model.h5'):
#         return None
#     return tf.keras.models.load_model('heart_disease_model.h5')

# model = load_my_model()

# # 3. UI Layout
# st.title("**_ECG Heart Disease Detection_**")
# st.write("Upload an ECG image to detect if the heart is **Normal** or shows signs of **Disease**.")

# # 4. File Uploader
# uploaded_file = st.file_uploader("Choose an ECG Image...", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     st.image(uploaded_file, caption='Uploaded ECG', width=300)
    
#     if model is None:
#         st.error("Model file 'heart_disease_model.h5' not found. Please run train.py first!")
#     else:
#         if st.button('Analyze ECG'):
#             with st.spinner('Analyzing...'):
#                 try:
#                     processed_image = preprocess_image(uploaded_file)
            
#                     prediction = model.predict(processed_image)
#                     probability = prediction[0][0]
                    
#                     if probability > 0.5:
#                         st.success(f"Result: **Normal** ")
#                         st.markdown("Go enjoy a chocolate!!")
#                     else:
#                         st.error(f"Result: **Disease Detected** ")
#                         st.markdown("Alert! Heart needs some TLC. Maybe less stress, more laughter")
#                 except Exception as e:
#                     st.error(f"Error processing image: {e}")


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
