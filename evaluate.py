"""
evaluate.py

This script evaluates the trained MobileNetV2 model on the unseen test dataset.
It computes basic accuracy/loss, generates a classification report, and displays
a confusion matrix for deeper insights into model performance.
"""

import os
# 1. Import TensorFlow, NumPy, Matplotlib, and Scikit-Learn
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# 2. Import test dataset and class names from preprocessing.py
try:
    from preprocessing import test_ds, class_names
except ImportError:
    print("Warning: Could not import test_ds and class_names from preprocessing.py")
    test_ds = None
    class_names = []

# Configuration Constants
MODEL_PATH = "models/best_skin_disease_model.keras"
RESULTS_DIR = "results"
REPORT_PATH = os.path.join(RESULTS_DIR, "classification_report.txt")
CM_PATH = os.path.join(RESULTS_DIR, "confusion_matrix.png")

def evaluate_test_dataset(model, dataset):
    """
    Evaluates the model natively using model.evaluate().
    """
    print("Evaluating model on the test dataset...")
    # 4. Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(dataset, verbose=1)
    
    # 5. Print Test Loss and Test Accuracy
    print("\n" + "="*40)
    print("TEST EVALUATION RESULTS")
    print("="*40)
    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("="*40 + "\n")
    return loss, accuracy

def generate_predictions(model, dataset):
    """
    Extracts true labels and generates predicted labels for the entire dataset.
    """
    print("Generating predictions for the test dataset...")
    y_true = []
    y_pred = []
    
    # 6. Generate predictions for the entire test dataset
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
        
    return np.array(y_true), np.array(y_pred)

def compute_advanced_metrics(y_true, y_pred):
    """
    Computes precision, recall, and f1 score.
    """
    # 7. Compute Accuracy, Precision, Recall, and F1 Score
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print("="*40)
    print("MACRO-AVERAGED METRICS")
    print("="*40)
    print(f"Accuracy Score: {acc:.4f}")
    print(f"Precision:      {prec:.4f}")
    print(f"Recall:         {rec:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print("="*40 + "\n")
    
    return acc, prec, rec, f1

def save_classification_report(y_true, y_pred, classes):
    """
    Generates, prints, and saves the scikit-learn classification report.
    """
    # 8. Generate a Classification Report using sklearn
    report = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    print("CLASSIFICATION REPORT:\n")
    print(report)
    
    # 10. Save: results/classification_report.txt
    with open(REPORT_PATH, "w") as f:
        f.write("Skin Disease Detection - Classification Report\n")
        f.write("="*60 + "\n")
        f.write(report)
    print(f"\nClassification report saved to: {REPORT_PATH}\n")

def save_and_plot_confusion_matrix(y_true, y_pred, classes):
    """
    Generates, saves, and visually displays the confusion matrix.
    """
    # 9. Generate a Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the display
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical', ax=plt.gca())
    plt.title('Confusion Matrix - Skin Disease Detection')
    plt.tight_layout()
    
    # 11. Save the confusion matrix image
    plt.savefig(CM_PATH)
    print(f"Confusion matrix image saved to: {CM_PATH}\n")
    
    # 12. Display the confusion matrix using Matplotlib
    print("Displaying Confusion Matrix (Close the window to terminate script)...")
    plt.show()

def main():
    """
    Main evaluation pipeline.
    """
    print("Initializing Evaluation Pipeline...\n")
    
    if test_ds is None or len(class_names) == 0:
        print("Error: test_ds or class_names could not be loaded. Please ensure dataset folders exist.")
        return
        
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at '{MODEL_PATH}'. Please ensure train.py ran successfully.")
        return
        
    # 13. Automatically create the "results" folder if it does not exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 3. Load the trained model
    print(f"Loading trained model from '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)
    
    # Evaluate Loss & Accuracy natively
    evaluate_test_dataset(model, test_ds)
    
    # Get True vs Predicted Labels
    y_true, y_pred = generate_predictions(model, test_ds)
    
    # Calculate Precision, Recall, F1
    compute_advanced_metrics(y_true, y_pred)
    
    # Text Report
    save_classification_report(y_true, y_pred, class_names)
    
    # Visual Confusion Matrix
    save_and_plot_confusion_matrix(y_true, y_pred, class_names)

# 14. Include a main() function so the file can be executed using `python evaluate.py`
if __name__ == "__main__":
    main()
