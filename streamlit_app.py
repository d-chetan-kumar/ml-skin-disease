"""
streamlit_app.py

AI-Based Skin Disease Detection using MobileNetV2
"""

import os
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -------------------------------
# Class Names (Same order as training)
# -------------------------------
class_names = [
    "Acne and Rosacea Photos",
    "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos",
    "Bullous Disease Photos",
    "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos",
    "Exanthems and Drug Eruptions",
    "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos",
    "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue Diseases",
    "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease",
    "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases",
    "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors",
    "Systemic Disease",
    "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives",
    "Vascular Tumors",
    "Vasculitis Photos",
    "Warts Molluscum and other Viral Infections"
]

MODEL_PATH = "models/best_skin_disease_model.keras"

st.set_page_config(
    page_title="AI Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    return keras_load_model(MODEL_PATH)

# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(uploaded_file):

    image = Image.open(uploaded_file).convert("RGB")

    resized = image.resize((224, 224))

    img = np.array(resized)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img, image

# -------------------------------
# Prediction
# -------------------------------
def predict(model, img):

    predictions = model.predict(img, verbose=0)

    class_index = np.argmax(predictions[0])

    confidence = float(predictions[0][class_index])

    disease = class_names[class_index]

    return disease, confidence
    # -------------------------------
# Main Application
# -------------------------------
def main():

    # Sidebar
    st.sidebar.title("Project Information")
    st.sidebar.markdown("**Model:** MobileNetV2 Transfer Learning")
    st.sidebar.markdown(f"**Classes:** {len(class_names)}")
    st.sidebar.markdown("**Framework:** TensorFlow + Streamlit")
    st.sidebar.markdown("**Dataset:** DermNet Dataset")
    st.sidebar.markdown("**Developer:** Student Project")

    # Header
    st.title("🩺 AI-Based Skin Disease Detection System")
    st.subheader(
        "Upload a skin disease image and let the trained MobileNetV2 model predict the disease category."
    )

    st.markdown("---")

    # Load Model
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Unable to load model.\n\n{e}")
        st.stop()

    # Upload Image
    uploaded_file = st.file_uploader(
        "Choose a skin image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is None:

        st.info("Please upload a skin image.")

    else:

        try:

            img_batch, original_image = preprocess_image(uploaded_file)

            col1, col2 = st.columns([1, 1.3])

            with col1:
                st.image(
                    original_image,
                    caption="Uploaded Image",
                    width=350
                )

            with col2:

                st.write("")
                st.write("")

                if st.button(
                    "🔍 Predict Disease",
                    use_container_width=True
                ):

                    with st.spinner("Analyzing image..."):

                        disease, confidence = predict(
                            model,
                            img_batch
                        )

                    st.success("Prediction Completed Successfully!")

                    st.markdown("## 🩺 Predicted Disease")

                    st.markdown(
                        f"""
### ✅ {disease}
"""
                    )

                    st.markdown("## 📊 Confidence")

                    st.progress(float(confidence))

                    st.metric(
                        "Model Confidence",
                        f"{confidence*100:.2f}%"
                    )

                    if confidence >= 0.90:
                        st.success(
                            "The model is highly confident about this prediction."
                        )

                    elif confidence >= 0.70:
                        st.info(
                            "The model is reasonably confident."
                        )

                    else:
                        st.warning(
                            "The confidence is low. Consider using another image with better lighting and clarity."
                        )

        except UnidentifiedImageError:

            st.error(
                "Invalid image. Please upload a valid JPG or PNG image."
            )

        except Exception as e:

            st.error(f"Prediction Error:\n\n{e}")

    st.markdown("---")

    with st.expander("ℹ About the Model"):

        st.write(
            """
This application uses **MobileNetV2 Transfer Learning**
trained on the **DermNet Dataset** containing **23 different
skin disease categories**.

### Workflow

1. Upload Image
2. Image resized to **224×224**
3. MobileNetV2 preprocessing
4. Deep Learning prediction
5. Disease name and confidence displayed

### Disclaimer

This application is developed only for educational and research
purposes. It should **NOT** be considered as a substitute for a
professional medical diagnosis.
"""
        )

    st.markdown("---")

    st.markdown(
        """
<div style="text-align:center">

### AI-Based Skin Disease Detection

Built using TensorFlow • Keras • MobileNetV2 • Streamlit

</div>
""",
        unsafe_allow_html=True,
    )


# -------------------------------
# Run Application
# -------------------------------
if __name__ == "__main__":
    main()
