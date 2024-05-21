import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

# Load the trained model
model_path = '../code/best_model_1st.keras'
model = load_model(model_path)

# Define the classes corresponding to the model's output
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# Function to preprocess the uploaded image
def proc_img(img, target_size=(128, 128)):
    img = img.resize(target_size)  # Resize to target size
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Define the Streamlit app layout
st.title('Sign Language Recognition')
uploaded_image = st.file_uploader("Choose a sign-language image...", type=["jpg", "jpeg", "png"])

# Perform predictions when an image is uploaded
if uploaded_image is not None:
    # Display the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_img = proc_img(img)

    # Ensure the input shape is correct for the model
    if processed_img.shape != (1, 128, 128, 3):
        st.write(f"Input shape is incorrect: {processed_img.shape}")
    else:
        st.write("Processed image shape:", processed_img.shape)
        
        # Make predictions
        predictions = model.predict(processed_img)
        st.write("Predictions:", predictions)

        # Find the predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes[predicted_class_index]

        st.write("Predicted Sign Language:", predicted_class)