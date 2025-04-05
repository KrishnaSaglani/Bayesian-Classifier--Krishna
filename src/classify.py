import numpy as np
import pandas as pd
import logging
import pickle
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Create results folder if not exists -------------------- #
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# -------------------- Logging -------------------- #
log_file = os.path.join(results_dir, "classification.log")
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

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

        predicted_class = max(class_probs, key=class_probs.get)
        predictions.append(predicted_class)

    return predictions

if __name__ == "__main__":
    # Load test features
    X_test, y_test = load_data("test_features.csv")
    logging.info("Loaded test data.")

    # Load model parameters
    with open("model_params.pkl", "rb") as f:
        model_params = pickle.load(f)

    class_priors = model_params['class_priors']
    means = model_params['means']
    variances = model_params['variances']
    unique_classes = model_params['unique_classes']

    logging.info("Loaded model parameters from 'model_params.pkl'.")

    # Predict
    y_pred = predict(X_test, class_priors, means, variances, unique_classes)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(unique_classes))

    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n" + report)

    # Save results.txt
    results_path = os.path.join(results_dir, "results.txt")
    with open(results_path, "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Save Confusion Matrix
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
    confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.show()
