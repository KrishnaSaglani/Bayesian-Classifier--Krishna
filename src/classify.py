import numpy as np
import pandas as pd
import logging
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_file):
    df = pd.read_csv(csv_file)
    labels = df['label'].values
    features = df.drop('label', axis=1).values
    return features, labels

def compute_mean_variance(features, labels):
    class_stats = {}
    class_data = defaultdict(list)

    for feature, label in zip(features, labels):
        class_data[label].append(feature)

    for label, vectors in class_data.items():
        vectors = np.array(vectors)
        mean = np.mean(vectors, axis=0)
        variance = np.var(vectors, axis=0) + 1e-6  # avoid zero division
        class_stats[label] = (mean, variance)

    return class_stats

def gaussian_probability(x, mean, var):
    exponent = - ((x - mean) ** 2) / (2 * var)
    coeff = 1.0 / np.sqrt(2 * np.pi * var)
    return coeff * np.exp(exponent)

def predict(features, class_stats, class_priors):
    predictions = []

    for x in features:
        class_probs = {}

        for label, (mean, var) in class_stats.items():
            log_probs = np.log(class_priors[label])
            log_probs += np.sum(np.log(gaussian_probability(x, mean, var)))
            class_probs[label] = log_probs

        predicted_class = max(class_probs, key=class_probs.get)
        predictions.append(predicted_class)

    return predictions

if __name__ == "__main__":
    logging.basicConfig(filename="classification.log", level=logging.INFO, format='%(asctime)s - %(message)s')

    # Load features
    X_train, y_train = load_data("train_features.csv")
    X_test, y_test = load_data("test_features.csv")

    logging.info("Loaded training and test data.")

    # Compute class statistics
    class_stats = compute_mean_variance(X_train, y_train)

    # Compute priors
    unique_classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    class_priors = {label: count / total for label, count in zip(unique_classes, counts)}

    # Predict
    y_pred = predict(X_test, class_stats, class_priors)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=sorted(unique_classes))

    # Print to console
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:")
    print(report)

    # Log
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("Classification Report:\n" + report)

    # Save results to a txt file
    with open("results.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion Matrix Visualization
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
    plt.savefig("confusion_matrix.png")
    plt.show()
