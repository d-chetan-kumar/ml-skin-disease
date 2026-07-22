"""
preprocessing.py

This module is responsible for loading, splitting, augmenting, and preprocessing
the DermNet dataset for a Skin Disease Detection model using MobileNetV2.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1 & 2. Define dataset paths & Configuration
# -----------------------------------------------------------------------------
TRAIN_DIR = "Dataset/train"
TEST_DIR = "Dataset/test"

BATCH_SIZE = 32         # 7. Use batch size = 32
IMG_SIZE = (224, 224)   # 6. Resize all images to 224x224 pixels
SEED = 42               # Consistent random seed for validation split

# -----------------------------------------------------------------------------
# 9. Sequential Data Augmentation Pipeline
# -----------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
], name="data_augmentation")

# 8. MobileNetV2 Preprocessing Function
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_train(image, label):
    """Applies augmentation and MobileNetV2 preprocessing to training data."""
    # 10. Apply augmentation ONLY to the training dataset
    # We specify training=True to ensure random transformations are applied
    image = data_augmentation(image, training=True)
    # 11. Apply MobileNetV2 preprocessing
    image = preprocess_input(image)
    return image, label

def preprocess_val_test(image, label):
    """Applies ONLY MobileNetV2 preprocessing to validation and test data."""
    # 11. Apply MobileNetV2 preprocessing
    image = preprocess_input(image)
    return image, label

def create_datasets():
    """Loads datasets from directory, splits, and applies preprocessing."""
    
    # 3 & 4. Load training dataset and automatically split (80% Train, 20% Val)
    print("Loading Training and Validation datasets...")
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )
    
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # 16. Read class names automatically from the dataset folders
    class_names = train_ds_raw.class_names

    # 5. Load the test dataset separately without any validation split
    print("Loading Testing dataset...")
    test_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    # 12. Optimize the data pipeline using cache() and prefetch()
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_dataset = train_ds_raw.map(preprocess_train, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    val_dataset = val_ds_raw.map(preprocess_val_test, num_parallel_calls=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    test_dataset = test_ds_raw.map(preprocess_val_test, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # 13. Print required information
    print("\n--- Dataset Information ---")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Training batches: {len(train_dataset)}")
    print(f"Validation batches: {len(val_dataset)}")
    print(f"Testing batches: {len(test_dataset)}")
    print("---------------------------\n")

    return train_dataset, val_dataset, test_dataset, class_names, train_ds_raw

def show_sample_batch(dataset, class_names):
    """
    14. Display one sample batch of training images using matplotlib.
    Uses the raw dataset (before MobileNetV2 preprocessing) so images display correctly.
    """
    print("Displaying a sample batch of training images...")
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(min(9, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Global Variables Export
# -----------------------------------------------------------------------------
# We initialize datasets so they can be imported directly in other scripts 
# like `train.py` (e.g., from preprocessing import train_ds, val_ds, test_ds)
train_ds = None
val_ds = None
test_ds = None
class_names = []
train_ds_raw = None

# We execute create_datasets() only if directories exist. This prevents import 
# errors when the dataset is not yet downloaded, while still allowing 
# datasets to be populated automatically if present.
if os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR):
    train_ds, val_ds, test_ds, class_names, train_ds_raw = create_datasets()
    
    if __name__ == "__main__":
        show_sample_batch(train_ds_raw, class_names)
else:
    print(f"Warning: Ensure '{TRAIN_DIR}' and '{TEST_DIR}' exist to load datasets.")