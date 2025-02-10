import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('cnn_model.h5', compile=False)

# Function to process the uploaded image
def process_image(img):
    img = img.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Title of the application
st.title('👶 Age Detection from Image 📸')
st.write("Upload a photo, and the model will predict the age.")

# Sidebar for additional interaction options
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Upload an image of a face.
2. The model will predict the age based on the image.
3. The output will be displayed below the image.
""")

# File uploader for the user to upload an image
file = st.file_uploader('Select an image (jpg, jpeg, png)', type=['jpg', 'jpeg', 'png'])

if file is not None:
    # Displaying the uploaded image
    img = Image.open(file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Process the image and predict the result
    image = process_image(img)
    prediction = model.predict(image)
    prediction = np.round(prediction).astype(int)  # Rounding the prediction

    # Show result in a more interactive format
    st.subheader("Prediction Result:")
    st.write(f"Predicted Age: **{prediction[0][0]}** years old")

    # Optionally, you can add a confidence message
    st.markdown(f"""
    **Confidence:** The model has made this prediction based on its trained data, but the prediction may vary depending on the quality of the image and other factors.
    """)

    # A divider for clarity
    st.markdown("---")
else:
    st.write("Please upload an image to get started.")
