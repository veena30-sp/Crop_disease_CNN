import streamlit as st
from huggingface_hub import hf_hub_download
import tensorflow as tf
from PIL import Image
import numpy as np

# App title and description
st.title("🌾 Crop Disease Detection")
st.write("Upload an image of a crop leaf to detect if it is Healthy, has Rust, or Powdery mildew.")

# Load the pre-trained model from Hugging Face
@st.cache_resource
def load_model():
    # Replace 'username/repo-name' and 'model.h5' with your Hugging Face repo details
    model_path = hf_hub_download(repo_id="veena30/DLproject_crop_disease", filename="model.h5")
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# Define the class labels
class_labels = ['Healthy', 'Rust', 'Powdery']

# Preprocess the uploaded image
def preprocess_image(image, target_size):
    image = image.resize(target_size)  # Resize to model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# File uploader for image upload
uploaded_image = st.file_uploader("Upload a crop leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Open and preprocess the image
    image = Image.open(uploaded_image)
    processed_image = preprocess_image(image, target_size=(224, 224))  # Adjust target size if needed

    # Make predictions
    prediction = model.predict(processed_image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display prediction results
    st.subheader("Prediction Results:")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Provide guidance based on predictions
    if predicted_class == "Healthy":
        st.success("The crop leaf appears to be healthy! 🌱")
    elif predicted_class == "Rust":
        st.warning("The crop leaf shows signs of rust disease. Consider consulting an agricultural expert.")
    elif predicted_class == "Powdery":
        st.warning("The crop leaf shows signs of powdery mildew. Take appropriate measures to manage it.")

# Footer
st.write("---")
st.write("This tool is designed for educational purposes and should not replace professional advice.")
