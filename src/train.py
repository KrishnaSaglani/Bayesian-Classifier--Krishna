# train.py - Train a Bayesian classifier from scratch (manual calculation)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load the features and labels
data = pd.read_csv('features.csv')
X = data.iloc[:, :-1].values  # Feature columns
y = data.iloc[:, -1].values   # Label column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Manually calculate class priors (P(y))
class_priors = {}
for c in np.unique(y_train):
    class_priors[c] = np.sum(y_train == c) / len(y_train)

# Manually calculate means and variances for each class
means = {}
variances = {}
for c in np.unique(y_train):
    X_c = X_train[y_train == c]
    means[c] = np.mean(X_c, axis=0)
    variances[c] = np.var(X_c, axis=0) + 1e-6  # Add small value to avoid division by zero

# Gaussian likelihood function (for each feature in a class)
def gaussian_likelihood(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))
