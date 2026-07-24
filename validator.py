"""
validator.py

This module implements a lightweight, modular validation layer for DermaVision.
It verifies whether an image belongs to the dermatology domain (skin/skin lesions)
using skin color distribution heuristics and pre-trained ImageNet classifier checks.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

# Global cache for the ImageNet model
_IMAGENET_MODEL = None

def get_imagenet_model():
    """
    Loads and caches the pre-trained ImageNet MobileNetV2 model.
    """
    global _IMAGENET_MODEL
    if _IMAGENET_MODEL is None:
        # Will load the cached weights from C:\Users\HP\.keras\models
        _IMAGENET_MODEL = MobileNetV2(weights='imagenet')
    return _IMAGENET_MODEL

def calculate_skin_percentage(img: Image.Image) -> float:
    """
    Calculates the percentage of pixels in the image that match human skin color ranges
    under RGB, YCbCr, and HSV color representations.
    """
    img_rgb = np.array(img.convert('RGB'))
    img_ycbcr = np.array(img.convert('YCbCr'))
    img_hsv = np.array(img.convert('HSV'))
    
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    y, cb, cr = img_ycbcr[:,:,0], img_ycbcr[:,:,1], img_ycbcr[:,:,2]
    h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
    
    # 1. RGB Rule
    rgb_rule = (r > 95) & (g > 40) & (b > 20) & \
               ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15) & \
               (np.abs(r - g) > 15) & (r > g) & (r > b)
               
    # 2. YCbCr Rule
    ycbcr_rule = (cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)
    
    # 3. HSV Rule (H range 0-50 corresponds to 0-35 on a 0-255 PIL scale)
    hsv_rule = (h <= 35) & (s >= 50) & (s <= 173) & (v >= 90)
    
    # Combined skin mask (union of all rules)
    combined_mask = rgb_rule | ycbcr_rule | hsv_rule
    return float(np.mean(combined_mask) * 100)

def is_valid_dermatology_image(img: Image.Image) -> bool:
    """
    Determines whether the given image is a valid dermatology image (skin/lesions).
    Returns True if valid, False otherwise.
    """
    # Step 1: Calculate skin color percentage
    skin_pct = calculate_skin_percentage(img)
    
    # If the image has less than 28.0% skin pixels, reject it immediately
    if skin_pct < 28.0:
        return False
        
    # Step 2: Run pre-trained ImageNet model to check for distinct everyday objects
    try:
        # Resize and preprocess for MobileNetV2
        img_resized = img.resize((224, 224))
        img_arr = np.array(img_resized)
        img_arr = preprocess_input(img_arr)
        img_batch = np.expand_dims(img_arr, axis=0)
        
        model = get_imagenet_model()
        preds = model.predict(img_batch, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]
        
        class_name = decoded[1].lower()
        confidence = float(decoded[2])
        
        # Determine confidence threshold dynamically based on skin percentage
        threshold = 0.80 if skin_pct > 75.0 else 0.20
        
        if confidence > threshold:
            # Check if the class name is skin-friendly/medical-adjacent
            allowed_keywords = [
                'band_aid', 'bandage', 'sunscreen', 'nipple', 'face_powder', 
                'lipstick', 'tick', 'hog', 'velvet', 'dough', 'sponge', 
                'nail', 'hair', 'hand', 'finger', 'foot', 'arm', 'leg', 
                'face', 'neck', 'body', 'spotlight', 'reflex_camera'
            ]
            is_allowed = any(keyword in class_name for keyword in allowed_keywords)
            if not is_allowed:
                return False
                
    except Exception as e:
        # If any error occurs, default to allowing the image (fallback to color check)
        pass
        
    return True
