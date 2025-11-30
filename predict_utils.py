# from PIL import Image, ImageOps
# import numpy as np

# def preprocess_image(uploaded_file):
#     """
#     Reads the uploaded Streamlit file, converts to PIL image,
#     resizes to 224x224, and normalizes it for the model.
#     """
#     # Open the image directly from the uploaded memory buffer
#     img = Image.open(uploaded_file).convert('RGB')
    
#     # Resize to match model input
#     img = img.resize((224, 224))
    
#     # Convert to numpy array
#     img_array = np.array(img)
    
#     # Add batch dimension (1, 224, 224, 3)
#     img_array = np.expand_dims(img_array, axis=0)
    
#     # Normalize pixel values
#     img_array = img_array / 255.0
    
#     return img_array

import cv2
import numpy as np
import tensorflow as tf

# Load TFLite Model
interpreter = tf.lite.Interpreter(model_path="heart_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_file):
    img = cv2.imread(image_file)
    img = cv2.resize(img, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_file):
    img = preprocess_image(image_file)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    prob = float(output[0][0])  # prediction probability

    return prob
