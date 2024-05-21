import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image

#model
# model_path = 'best_model_1st.keras'
# model = load_model(model_path)

#classes for model's output
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']

# preprocess the uploaded image
def proc_img(img, target_size=(128, 128)):
    img = img.resize(target_size)  # re-sizing image to target size
    img_array = image.img_to_array(img)  # converting image to array
    img_array = np.expand_dims(img_array, axis=0)  # adding batch dimension
    img_array = img_array / 255.0  # normalizing the image
    return img_array

# Streamlit app layout
st.title('Sign Language Recognition')
uploaded_image = st.file_uploader("Choose a sign-language image...", type=["jpg", "jpeg", "png"])

#model
model_path = 'best_model_1st.keras'
model = load_model(model_path)

# predictions when an image is uploaded
if uploaded_image is not None:
    # displaying the uploaded image
    img = Image.open(uploaded_image)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # preprocessing the image with the proc_img function above
    processed_img = proc_img(img)

    # ensure that input shape is correct for the model
    if processed_img.shape != (1, 128, 128, 3):
        st.write(f"Input shape is incorrect: {processed_img.shape}")
    else:
        st.write("Processed image shape:", processed_img.shape)
        
        # predictions
        predictions = model.predict(processed_img)
        st.write("Predictions:", predictions)

        # predicted class index
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class = classes[predicted_class_index]

        st.write("Predicted Sign Language:", predicted_class)
