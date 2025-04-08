# src/classify.py

import numpy as np
import pandas as pd
import logging
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image, ImageDraw, ImageFont

def visualize_predictions(image_dir="fruits-360/custom_images", label_file="custom_labels.txt", predictions=[], actual_labels=[]):
    output_dir = "results_custom/visualized"
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Read filename -> actual label mapping from file
    filename_to_actual = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                filename, actual_label = parts
                filename_to_actual[filename] = actual_label

    # Step 2: Sort filenames to match order of predictions
    filenames = list(filename_to_actual.keys())

    for i, filename in enumerate(filenames):
        image_path = os.path.join(image_dir, filename)
        try:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)

            pred = predictions[i]
            actual = filename_to_actual[filename]
            label_text = f"Predicted: {pred} | Actual: {actual}"

            font = ImageFont.load_default()

            # Draw white rectangle for text
            text_width = draw.textlength(label_text, font=font)
            draw.rectangle([0, 0, text_width + 10, 20], fill="white")
            draw.text((5, 5), label_text, fill="black", font=font)

            out_path = os.path.join(output_dir, f"{i+1}_{filename}")
            image.save(out_path)

        except Exception as e:
            print(f"Error processing {filename}: {e}")


def classify_custom_images():
    os.makedirs("results_custom", exist_ok=True)

    # Set up custom logger
    logger = logging.getLogger("FruitClassifier")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        file_handler = logging.FileHandler("results_custom/classification.log")
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    def load_data(csv_file):
        df = pd.read_csv(csv_file)
        labels = df['label'].values
        features = df.drop('label', axis=1).values
        return features, labels

    def gaussian_probability(x, mean, var):
        exponent = - ((x - mean) ** 2) / (2 * var)
        coeff = 1.0 / np.sqrt(2 * np.pi * var)
        return coeff * np.exp(exponent)

    def predict(features, class_priors, means, variances, unique_classes):
        predictions = []
        for x in features:
            class_probs = {}
            for label in unique_classes:
                log_probs = np.log(class_priors[label] + 1e-9)
                log_probs += np.sum(np.log(gaussian_probability(x, means[label], variances[label]) + 1e-9))
                class_probs[label] = log_probs
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions

    # Load features from PCA-processed CSV
    X_test, _ = load_data("features/custom_adv_pca.csv")
    logger.info("Loaded test data.")

    # Load model
    with open("model/model_parameters.pkl", "rb") as f:
        model_params = pickle.load(f)

    class_priors = model_params['class_priors']
    means = model_params['means']
    variances = model_params['variances']
    unique_classes = model_params['unique_classes']

    # Predict
    y_pred = predict(X_test, class_priors, means, variances, unique_classes)

    # ✅ Load correct actual labels from filenames
    actual_labels = []
    with open("custom_labels.txt", "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 2:
                actual_labels.append(parts[1])

    # Metrics
    acc = accuracy_score(actual_labels, y_pred)
    report = classification_report(actual_labels, y_pred, zero_division=0)
    cm = confusion_matrix(actual_labels, y_pred, labels=sorted(unique_classes))

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + report)

    with open("results_custom/results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=sorted(unique_classes),
                yticklabels=sorted(unique_classes))
    plt.title("Confusion Matrix - Fruit Classification")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("results_custom/confusion_matrix.png")
    plt.show()

    # 🔍 Visualize and save predictions on images
    visualize_predictions(predictions=y_pred, actual_labels=actual_labels)

if __name__ == "__main__":
    classify_custom_images()

