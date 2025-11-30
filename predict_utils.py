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

