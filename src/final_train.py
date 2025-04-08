# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import logging
import pickle
import os

def train():
    os.makedirs("model", exist_ok=True)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
        logging.FileHandler("model/train.log"),
        logging.StreamHandler()
    ])

    data = pd.read_csv('features/train_adv_pca.csv')
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Training started...")

    def calculate_class_priors(labels):
        priors = {}
        total = len(labels)
        for label in np.unique(labels):
            priors[label] = np.sum(labels == label) / total
        return priors
    
    class_priors = calculate_class_priors(y_train)
    for label, prior in class_priors.items():
        logging.info(f"Class '{label}' prior: {prior:.4f}")


    def calculate_means_and_variances(X, y):
        means = {}
        variances = {}
        for label in np.unique(y):
            class_data = X[y == label]
            means[label] = np.mean(class_data, axis=0)
            variances[label] = np.var(class_data, axis=0) + 1e-4
        return means, variances

    means, variances = calculate_means_and_variances(X_train, y_train)

    def gaussian_likelihood(x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(X, class_priors, means, variances, unique_classes):
        predictions = []
        total_samples = len(X)
        logging.info(f"Making predictions on {total_samples} test samples...")
        start_time = time.time()

        for idx, sample in enumerate(X):
            class_probabilities = {}
            for label in unique_classes:
                prior = np.log(class_priors[label] + 1e-9)
                likelihood = np.sum(np.log(gaussian_likelihood(sample, means[label], variances[label]) + 1e-9))
                class_probabilities[label] = prior + likelihood

            best_label = max(class_probabilities, key=class_probabilities.get)
            predictions.append(best_label)

            if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
                logging.info(f"Predicted {idx + 1}/{total_samples} samples... ({(idx + 1) / total_samples * 100:.2f}%)")


        logging.info(f"Prediction completed in {time.time() - start_time:.2f} seconds.")
        return np.array(predictions)

    y_pred = predict(X_test, class_priors, means, variances, np.unique(y_train))

    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
    logging.info(f"\n{classification_report(y_test, y_pred)}")

    model_params = {
        'class_priors': class_priors,
        'means': means,
        'variances': variances,
        'unique_classes': np.unique(y_train)
    }

    with open('model/model_parameters.pkl', 'wb') as f:
        pickle.dump(model_params, f)

    logging.info("Model parameters saved to 'model/model_parameters.pkl'")

if __name__ == "__main__":
    train()
