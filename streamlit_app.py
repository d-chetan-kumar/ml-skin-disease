"""
streamlit_app.py

This script implements a production-quality Streamlit web application
for the AI-Based Skin Disease Detection System. It allows users to upload
an image, preprocesses it, and uses a trained MobileNetV2 model to predict
the disease category.
"""

import os
# 1. Import required dependencies
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
# Import load_model and preprocess_input precisely as required
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# 1. Import class_names from preprocessing.py safely
try:
    from preprocessing import class_names
except ImportError:
    st.error("Warning: Could not import class_names from preprocessing.py")
    class_names = []

# Configuration Constant
MODEL_PATH = "models/best_skin_disease_model.keras"

# 3. Set the page configuration (Must be the first Streamlit command)
st.set_page_config(
    page_title="AI Skin Disease Detection",
    page_icon="🩺",
    layout="wide"
)

# 19. Modular function: load_model()
# 2. Load the trained model only once using @st.cache_resource
@st.cache_resource
def load_model():
    """
    Loads the compiled Keras model from the specified path.
    Caches the model to prevent reloading on every interaction.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at '{MODEL_PATH}'.")
    return keras_load_model(MODEL_PATH)


# 19. Modular function: preprocess_image()
def preprocess_image(uploaded_file):
    """
    Validates, loads, resizes, and preprocesses an uploaded image for MobileNetV2.
    """
    # Load the image using PIL
    img = Image.open(uploaded_file).convert('RGB')
    
    # 9. Resize exactly like predict.py: 224 x 224
    img_resized = img.resize((224, 224))
    
    # 9. Convert to NumPy array
    img_array = np.array(img_resized)
    
    # 9. Apply MobileNetV2 preprocessing (preprocess_input)
    img_array = preprocess_input(img_array)
    
    # 9. Expand dimensions for batch formatting (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, img


# 19. Modular function: predict()
def predict(model, img_batch, classes):
    """
    Predicts the disease using the model and returns the class name and confidence score.
    """
    # 11. Predict using model.predict()
    preds = model.predict(img_batch, verbose=0)
    
    class_idx = np.argmax(preds[0])
    
    if classes and len(classes) > class_idx:
        disease_name = classes[class_idx]
    else:
        disease_name = f"Class Index {class_idx}"
        
    confidence = float(preds[0][class_idx])
    
    return disease_name, confidence


# 19. Modular function: main()
def main():
    """
    Main Streamlit application pipeline. Handles UI structure, user interactions,
    and exception handling.
    """
    
    # 5. Sidebar Configuration
    st.sidebar.title("Project Information")
    st.sidebar.markdown("**Model:**\nMobileNetV2 Transfer Learning")
    st.sidebar.markdown("**Classes:**\n23")
    st.sidebar.markdown("**Framework:**\nTensorFlow + Streamlit")
    st.sidebar.markdown("**Dataset:**\nDermNet Dataset")
    st.sidebar.markdown("**Developer:**\nStudent Project")
    
    # 4. Design a professional UI (Top Section)
    st.title("🩺 AI-Based Skin Disease Detection System")
    st.subheader("Upload a skin disease image and let the trained MobileNetV2 model predict the disease category.")
    st.markdown("---")
    
    # 18. Exception handling for model loading
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # 6. Allow image upload using st.file_uploader() (jpg, jpeg, png)
    uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is None:
        # 7. When no image is uploaded display prompt message
        st.info("Please upload a skin image.")
    else:
        # Define layout using columns for side-by-side presentation
        col1, col2 = st.columns([1, 1.5])
        
        # 18. Exception handling for invalid images
        try:
            # Execute Preprocessing
            img_batch, original_img = preprocess_image(uploaded_file)
            
            with col1:
                # 8. Display image on the left side, width around 350 pixels
                st.image(original_img, caption="Uploaded Image", width=350)
                
            with col2:
                st.write("") # Padding
                
                # 10. Add a large centered button
                _, btn_col, _ = st.columns([1, 2, 1])
                with btn_col:
                    predict_button = st.button("Predict Disease", use_container_width=True)
                
                # 10. Prediction should only happen after pressing the button
                if predict_button:
                    with st.spinner("Analyzing Image..."):
                        # 18. Exception handling for Prediction errors
                        try:
                            # 11. Execute prediction pipeline
                            disease_name, confidence = predict(model, img_batch, class_names)
                            
                            # 12 & 13. Show prediction inside a green success box
                            st.success(f"""
                            **Predicted Disease**  
                            ### {disease_name}  
                            
                            **Confidence**  
                            ### {confidence * 100:.2f}%
                            """)
                            
                            # 14. Create a confidence bar using st.progress()
                            st.progress(confidence)
                            
                            # 15. Under prediction display: Model Confidence XX.XX%
                            st.caption(f"Model Confidence: {confidence * 100:.2f}%")
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                            
        except UnidentifiedImageError:
            st.error("Error: Unsupported image format or corrupted file. Please provide a valid JPG/PNG.")
        except Exception as e:
            st.error(f"Error processing image: {e}")
            
    st.markdown("---")
    
    # 16. Add an expandable section: About the Model
    with st.expander("About the Model"):
        st.write("""
        This application uses MobileNetV2 Transfer Learning trained on the DermNet dataset containing 23 skin disease categories.
        
        **Disclaimer:** This prediction is intended only for educational and research purposes and should not replace professional medical diagnosis.
        """)
        
    # 17. Add a footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px;">
            <hr>
            <p style="color: grey; font-weight: bold;">AI-Based Skin Disease Detection</p>
            <p style="color: grey;">Built using TensorFlow, Keras and Streamlit</p>
            <hr>
        </div>
        """,
        unsafe_allow_html=True
    )


# 21. The file should execute using `streamlit run streamlit_app.py`
if __name__ == "__main__":
    main()
