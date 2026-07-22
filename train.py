"""
train.py

This script loads the preprocessed datasets and the compiled MobileNetV2 model,
trains the model using transfer learning, applies appropriate callbacks, 
saves the best and final models, and generates training performance graphs.
"""

import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# 2. Import training and validation datasets from preprocessing.py
try:
    from preprocessing import train_ds, val_ds
except ImportError:
    print("Warning: Could not import datasets from preprocessing.py")
    train_ds = None
    val_ds = None

# Configuration Constants
EPOCHS = 20
MODEL_PATH = "models/skin_disease_model.keras"
BEST_MODEL_PATH = "models/best_skin_disease_model.keras"
FINAL_MODEL_PATH = "models/final_skin_disease_model.keras"
RESULTS_DIR = "results"

def setup_callbacks():
    """
    Sets up EarlyStopping and ModelCheckpoint callbacks.
    """
    # 4. Use callbacks: EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    # 4. Use callbacks: ModelCheckpoint
    model_checkpoint = ModelCheckpoint(
        filepath=BEST_MODEL_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    return [early_stopping, model_checkpoint]

def plot_and_save_history(history):
    """
    Plots training/validation accuracy and loss, and saves them to the results/ folder.
    """
    # 13. Automatically create the "results" folder if it does not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Extract metrics from history
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(1, len(acc) + 1)
    
    # 9. Plot Training & Validation Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    # 10. Save the accuracy graph
    acc_path = os.path.join(RESULTS_DIR, 'accuracy.png')
    plt.savefig(acc_path)
    plt.close()
    
    # 9. Plot Training & Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    # 10. Save the loss graph
    loss_path = os.path.join(RESULTS_DIR, 'loss.png')
    plt.savefig(loss_path)
    plt.close()
    
    print(f"\nGraphs successfully saved to:\n- {acc_path}\n- {loss_path}")

def print_final_statistics(history):
    """
    Prints the final accuracy and loss values after training finishes.
    """
    # 11. Print Final Training and Validation Statistics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print("\n" + "="*40)
    print("FINAL TRAINING STATISTICS")
    print("="*40)
    print(f"Final Training Accuracy:   {final_train_acc:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.4f}")
    print(f"Final Training Loss:       {final_train_loss:.4f}")
    print(f"Final Validation Loss:     {final_val_loss:.4f}")
    print("="*40 + "\n")

def main():
    """
    Main execution pipeline to load data, load model, train, save, and plot.
    """
    print("Initializing Training Pipeline...\n")
    
    # Check if datasets loaded properly
    if train_ds is None or val_ds is None:
        print("Error: Datasets are missing. Please verify 'Dataset/train' exists and preprocessing.py works.")
        return
        
    # Check if compiled model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'. Please run model.py first.")
        return
        
    # 13. Ensure 'results' directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 3. Load the compiled model
    print(f"Loading compiled model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    
    # Get callbacks setup
    callbacks = setup_callbacks()
    
    # 5. Train the model using model.fit()
    # 6. Use epochs = 20
    # 7. Display training progress for every epoch (handled natively by verbose=1 in fit)
    print(f"\nStarting model training for up to {EPOCHS} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # 8. Save the training history & 9-10. Plot/Save graphs
    plot_and_save_history(history)
    
    # 11. Print final stats
    print_final_statistics(history)
    
    # 12. Save the final trained model
    model.save(FINAL_MODEL_PATH)
    print(f"✔ Success: Final trained model saved to '{FINAL_MODEL_PATH}'")

# 14. Include a main() function so the file can be executed directly
if __name__ == "__main__":
    main()
