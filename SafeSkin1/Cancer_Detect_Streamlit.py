import streamlit as st
from PIL import Image
# import tensorflow as tf
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np
from tensorflow.keras.models import load_model



x_train_mean = 159.8
x_train_std = 46.7



# # Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 75))
    img_array = np.array(img)
    img_array = (img_array - x_train_mean) / x_train_std
    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)
    return img_array

# # Function to make predictions
def make_prediction(model, img_array):
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions

def decode_predictions(predictions):
    label = np.argmax(predictions)

    labels = ["Melanocytic nevi", "Melanoma", "Benign keratosis-like lesions", "Basal cell carcinoma", "Actinic keratoses", "Vascular lesions" ,"Dermatofibroma"]
    return labels[label]


model_path = 'model.hdf5'
model = load_model(model_path)


# Streamlit app
st.image('logo png 100.png')
st.title("NMU - Skin Diseases Detection")

st.title("Skin Diseases Detection Web App")
st.write("Upload an image and click the button to detect the disease.")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Display the uploaded image
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    #st.write("Classifying...")

    # Make predictions when the button is clicked
    if st.button("Classify Image"):
        # Preprocess the uploaded image
        img_array = load_and_preprocess_image(uploaded_file)

        # # Make predictions using the model
        predictions = make_prediction(model, img_array)
        
        
        st.write(f" Prediction is : {predictions}")
        st.write(f" Accuracy: 77.0344%")


