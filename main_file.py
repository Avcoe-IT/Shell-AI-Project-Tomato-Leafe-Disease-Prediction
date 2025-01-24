import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Ensure page configuration is the first Streamlit command
st.set_page_config(page_title="Tomato Leaf Disease Classifier", page_icon="🍅", layout="wide")

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model('tomato_disease (1).h5')  # Ensure this path is correct
    return model

model = load_trained_model()

# Image preprocessing function
IMG_SIZE = (224, 224)  # Same size used during training

def preprocess_image(image):
    img = image.resize(IMG_SIZE)  # Resize image
    img_array = img_to_array(img)  # Convert image to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit App UI
st.markdown(
    """
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .main-heading {
            text-align: center;
            color: #007f5f;
        }
        .upload-section, .how-it-works {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        .footer {
            text-align: center;
            color: gray;
            margin-top: 50px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='main-heading'>🍅 Tomato Leaf Disease Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 16px;'>Upload a clear image of a tomato plant leaf to determine its health status.</p>",
    unsafe_allow_html=True,
)

# Create layout sections
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
    st.markdown("### 📤 Upload an Image")
    uploaded_file = st.file_uploader(
        "Choose a tomato leaf image (JPG, JPEG, PNG)...",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict button
        if st.button("🌟 Classify"):
            with st.spinner("Analyzing the image..."):
                # Preprocess and predict
                image = load_img(uploaded_file)
                processed_image = preprocess_image(image)
                prediction = model.predict(processed_image)

                # Class labels (ensure these match your model's output classes)
                class_labels = [
                    'Tomato__Bacterial_spot',
                    'Tomato__Early_blight',
                    'Tomato__healthy',
                    'Tomato__Late_blight',
                    'Tomato__Leaf_Mold',
                    'Tomato__Septoria_leaf_spot',
                    'Tomato__Spider_mites Two-spotted_spider_mite',
                    'Tomato__Target_Spot',
                    'Tomato__Tomato_mosaic_virus',
                    'Tomato__Yellow_Leaf_Curl_Virus'
                ]

                # Get predicted class and confidence
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_label = class_labels[predicted_class_index]
                confidence = prediction[0][predicted_class_index] * 100

                # Display prediction and confidence
                st.success(f"🌿 **Predicted Class:** {predicted_label}")
                st.info(f"📊 **Confidence Level:** {confidence:.2f}%")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='how-it-works'>", unsafe_allow_html=True)
    st.markdown("### 📚 How It Works")
    st.write(
     """
    1. Upload a **clear and focused image** of the tomato plant leaf.
    2. Click the **Classify** button to process the image.
    3. The model will analyze the image and predict its health status:
        - **Bacterial Spot**: Identifies bacterial infections on leaves.
        - **Early Blight**: Detects early signs of blight disease.
        - **Healthy**: Confirms the plant shows no signs of disease.
        - **Late Blight**: Identifies advanced stages of blight disease.
        - **Leaf Mold**: Recognizes the presence of mold on leaves.
        - **Septoria Leaf Spot**: Detects fungal spots caused by Septoria.
        - **Spider Mites**: Identifies damage caused by spider mites.
        - **Target Spot**: Detects symptoms of fungal target spots.
        - **Mosaic Virus**: Detects patterns of damage from the virus.
        - **Yellow Leaf Curl Virus**: Identifies symptoms of this viral disease.
    4. Ensure proper lighting and clarity for accurate predictions.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<footer class='footer'>Powered by <b>TensorFlow</b> & <b>Streamlit</b></footer>",
    unsafe_allow_html=True,
)
