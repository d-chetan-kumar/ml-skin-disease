"""
predict.py

This script allows users to input an image path, loads the trained MobileNetV2
model, applies proper preprocessing, makes a prediction, and displays the result
both in the terminal and using Matplotlib.
"""

import os
# 1. Import TensorFlow, NumPy, Matplotlib, PIL (Pillow), and os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError

# 2. Import class_names from preprocessing.py
try:
    from preprocessing import class_names
except ImportError:
    print("Warning: Could not import class_names from preprocessing.py")
    class_names = []

# Configuration Constants
MODEL_PATH = "models/best_skin_disease_model.keras"

def load_trained_model(model_path):
    """
    Loads the compiled Keras model from the specified path.
    Raises an error if the model does not exist.
    """
    # 17. Exception handling for model loading
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'. Please train the model first.")
    
    print(f"Loading trained model from '{model_path}'...")
    return load_model(model_path)


def preprocess_image(image_path):
    """
    Validates, loads, resizes, and preprocesses an image for MobileNetV2.
    Returns both the preprocessed batch and the original PIL image for display.
    """
    # 5. Verify that the image exists
    # 17. Exception handling for invalid image path
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"The image path '{image_path}' does not exist.")
    
    # 6. Load the image using PIL
    # 17. UnidentifiedImageError will be caught in main for unsupported formats
    img = Image.open(image_path).convert('RGB')
    
    # 7. Resize the image to 224x224
    img_resized = img.resize((224, 224))
    
    # 8. Convert the image into a NumPy array
    img_array = np.array(img_resized)
    
    # 9. Apply MobileNetV2 preprocessing
    img_array = preprocess_input(img_array)
    
    # 10. Expand the batch dimension before prediction (shape becomes (1, 224, 224, 3))
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch, img


def predict_image(model, img_batch, classes):
    """
    Predicts the disease using the model and returns the class name and confidence.
    """
    # 11. Predict the disease using model.predict()
    preds = model.predict(img_batch, verbose=0)
    
    # 12. Determine Predicted class index, disease name, and confidence score
    class_idx = np.argmax(preds[0])
    
    if classes and len(classes) > class_idx:
        disease_name = classes[class_idx]
    else:
        # Fallback if class_names is empty or mismatched
        disease_name = f"Class Index {class_idx}"
        
    confidence = preds[0][class_idx] * 100.0
    
    return disease_name, confidence


def display_prediction(img, disease_name, confidence):
    """
    Displays the prediction result in the terminal and using a Matplotlib window.
    """
    # 13. Display terminal output in specified format
    print("\n" + "-"*36)
    print("Prediction Result")
    print("-"*36)
    print("Predicted Disease :")
    print(disease_name)
    print(f"\nConfidence :")
    print(f"{confidence:.2f}%")
    print("-"*36 + "\n")
    
    # 14. Display the uploaded image using Matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    
    # 15. Set the image title
    plt.title(f"Predicted:\n{disease_name}\n\nConfidence:\n{confidence:.2f}%", pad=20, fontsize=14)
    
    # 16. Remove image axes
    plt.axis('off')
    
    # Render
    plt.tight_layout()
    plt.show()


def main():
    """
    Main execution pipeline for interactive prediction.
    """
    # Attempt to load the model
    try:
        model = load_trained_model(MODEL_PATH)
    except Exception as e:
        print(f"Error Loading Model: {e}")
        return

    # 4. Accept an input image path from the user
    image_path = input("Enter image path: ").strip()
    
    # Remove quotes if user dragged and dropped the file into terminal
    if image_path.startswith(('"', "'")) and image_path.endswith(('"', "'")):
        image_path = image_path[1:-1]
    
    # Attempt to load and preprocess the image
    try:
        img_batch, original_img = preprocess_image(image_path)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return
    except UnidentifiedImageError:
        print("\nError: Unsupported image format or corrupted file. Please provide a valid JPG/PNG.")
        return
    except Exception as e:
        print(f"\nError processing image: {e}")
        return
        
    # Attempt prediction and display
    try:
        disease_name, confidence = predict_image(model, img_batch, class_names)
        display_prediction(original_img, disease_name, confidence)
    except Exception as e:
        print(f"\nError during prediction: {e}")


# 21. The file should execute using python predict.py
if __name__ == "__main__":
    main()
