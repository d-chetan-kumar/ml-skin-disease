"""
model.py

This module is responsible for defining, building, and compiling the MobileNetV2
transfer learning model for Skin Disease Detection. It saves the untrained compiled
model to the 'models/' directory.
"""

import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# 2. Import the training datasets (or ensure compatibility with preprocessing.py)
# Note: If dataset folders don't exist yet, these will be None/Empty as handled in preprocessing.py.
try:
    from preprocessing import train_ds, val_ds, test_ds, class_names
except ImportError:
    print("Warning: preprocessing.py not found or unable to import datasets.")
    class_names = []

def build_transfer_model(input_shape=(224, 224, 3), num_classes=23):
    """
    Builds the MobileNetV2 transfer learning model using the Functional API.
    
    Args:
        input_shape (tuple): Shape of the input images.
        num_classes (int): Number of output classes for the final Dense layer.
        
    Returns:
        base_model (Model): The pre-trained MobileNetV2 base model.
        model (Model): The complete custom classification model.
    """
    
    # 3. Create a MobileNetV2 model (weights="imagenet", include_top=False, input_shape=(224,224,3))
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # 4. Freeze all layers of the MobileNetV2 base model
    base_model.trainable = False

    # 6. Construct the complete transfer learning model using the Functional API
    inputs = Input(shape=input_shape, name="input_image")
    
    # We pass training=False to base_model to ensure BatchNormalization layers
    # run in inference mode and don't update their statistics.
    x = base_model(inputs, training=False)
    
    # 5. Build a custom classification head
    x = GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = Dropout(0.3, name="dropout_0.3")(x)
    
    # Final dense layer with Softmax activation
    outputs = Dense(num_classes, activation="softmax", name="classification_head")(x)

    model = Model(inputs=inputs, outputs=outputs, name="skin_disease_mobilenetv2")

    return base_model, model


def compile_model(model, learning_rate=0.0001):
    """
    Compiles the Keras model with Adam optimizer, SparseCategoricalCrossentropy, and accuracy.
    """
    # 7. Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    return model


def main():
    """Main execution function to build, compile, summarize, and save the model."""
    
    print("Initializing MobileNetV2 Model Building Process...\n")
    
    # Determine number of classes (default to 23 as requested)
    num_classes = len(class_names) if class_names else 23
    print(f"Configuring model for {num_classes} classes...")

    # Build the model
    base_model, model = build_transfer_model(input_shape=(224, 224, 3), num_classes=num_classes)
    
    # Compile the model
    model = compile_model(model)

    # 8. Print Base model summary
    print("\n" + "="*50)
    print("BASE MODEL SUMMARY (MobileNetV2)")
    print("="*50)
    base_model.summary()
    
    # 8. Print Complete model summary
    print("\n" + "="*50)
    print("COMPLETE TRANSFER LEARNING MODEL SUMMARY")
    print("="*50)
    model.summary()
    
    # 8. Print Parameter Information Explicitly
    # Keras models have methods to fetch these values natively.
    import numpy as np
    trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_count = trainable_count + non_trainable_count

    print("\n" + "="*50)
    print("PARAMETER COUNT INFO")
    print("="*50)
    print(f"Total parameters:         {int(total_count):,}")
    print(f"Trainable parameters:     {int(trainable_count):,}")
    print(f"Non-trainable parameters: {int(non_trainable_count):,}")
    print("="*50 + "\n")

    # 9. Automatically create a folder named models/ if it does not exist
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    print(f"Checked/Created directory: '{model_dir}/'")

    # 10. Save the compiled model as models/skin_disease_model.keras
    save_path = os.path.join(model_dir, "skin_disease_model.keras")
    model.save(save_path)
    
    print(f"\n✔ Success: Model successfully compiled and saved to '{save_path}'")
    print("Training has not been started.")

# 11. Include a main() function so the file can be executed directly
if __name__ == "__main__":
    main()
