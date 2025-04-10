# src/classify.py

import numpy as np
import pandas as pd
import logging
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def classify():
    os.makedirs("results2", exist_ok=True)

    # Set up custom logger
    logger = logging.getLogger("FruitClassifier")
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers on repeated runs
    if not logger.handlers:
        file_handler = logging.FileHandler("results2/classification.log")
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

    X_test, y_test = load_data("features/test_adv_pca.csv")
    logger.info("Loaded test data.")

    with open("model/model_parameters.pkl", "rb") as f:
        model_params = pickle.load(f)

    class_priors = model_params['class_priors']
    means = model_params['means']
    variances = model_params['variances']
    unique_classes = model_params['unique_classes']
    

    y_pred = predict(X_test, class_priors, means, variances, unique_classes)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(unique_classes))

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("Classification Report:\n" + report)

    with open("results2/results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

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
    plt.savefig("results2/confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    classify()
