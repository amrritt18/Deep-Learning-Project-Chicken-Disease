import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = "artifacts/training/vgg16_finetuned.h5"
IMAGE_SIZE = (224, 224)
CLASS_NAMES = {
    0: "Coccidiosis",
    1: "Healthy"
}


@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(model, uploaded_image):
    image = uploaded_image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    probabilities = model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_class = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_class])

    return (
        CLASS_NAMES[predicted_class],
        confidence,
        probabilities
    )


st.set_page_config(
    page_title="Chicken Disease Classifier",
    page_icon="🐔",
    layout="centered"
)


st.title("Chicken Fecal Disease Classifier")
st.write(
    "Upload a chicken fecal image to classify it as "
    "**Healthy** or **Coccidiosis**."
)


try:
    model = load_trained_model()
    st.success("Model loaded successfully.")

except FileNotFoundError:
    st.error(
        "Trained model not found. Please train the model first."
    )
    st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(
        image,
        caption="Input fecal image",
        use_container_width=True
    )

    if st.button("Predict Disease"):
        with st.spinner("Analyzing image..."):
            prediction, confidence, probabilities = predict_image(
                model,
                image
            )

        st.subheader("Prediction")

        if prediction == "Coccidiosis":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.write(
            f"Confidence: **{confidence * 100:.2f}%**"
        )

        st.subheader("Class Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Coccidiosis",
                f"{probabilities[0] * 100:.2f}%"
            )

        with col2:
            st.metric(
                "Healthy",
                f"{probabilities[1] * 100:.2f}%"
            )

        st.progress(confidence)