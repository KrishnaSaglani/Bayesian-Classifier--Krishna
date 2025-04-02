import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import logging

# Setup logging to file and console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("train.log"),
    logging.StreamHandler()
])

# Load the features and labels
data = pd.read_csv('features.csv')
X = data.iloc[:, :-1].values  # Feature columns
y = data.iloc[:, -1].values   # Label column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log progress
logging.info("Training started...")

# Manually calculate class priors (P(y))
class_priors = {}
start_time = time.time()  # Start time for logging

logging.info(f"Calculating class priors...")
for c in np.unique(y_train):
    class_priors[c] = np.sum(y_train == c) / len(y_train)
    logging.info(f"Calculated prior for class '{c}': {class_priors[c]:.4f}")

priors_calculation_time = time.time() - start_time
logging.info(f"Class priors calculation completed in {priors_calculation_time:.2f} seconds.\n")

# Manually calculate means and variances for each class
means = {}
variances = {}
logging.info(f"Calculating means and variances for each class...")

start_time = time.time()
for c in np.unique(y_train):
    X_c = X_train[y_train == c]
    means[c] = np.mean(X_c, axis=0)
    variances[c] = np.var(X_c, axis=0) + 1e-6  # Add small value to avoid division by zero
    logging.info(f"Class '{c}' - Mean: {means[c][:3]}... Variance: {variances[c][:3]}...")  # Log only first 3 features for brevity

means_variance_time = time.time() - start_time
logging.info(f"Means and variances calculation completed in {means_variance_time:.2f} seconds.\n")

# Gaussian likelihood function (for each feature in a class)
def gaussian_likelihood(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))

# Manually calculate posterior probabilities (P(y|x)) for each test instance
def predict(X):
    predictions = []
    total_samples = X.shape[0]
    
    logging.info(f"Making predictions on {total_samples} test samples...")
    
    start_time = time.time()
    
    for idx, x in enumerate(X):
        class_probs = {}
        for c in np.unique(y_train):
            # Log of prior (P(y))
            prior = np.log(class_priors[c])
            
            # Log of likelihood (P(x|y)) for each feature
            likelihood = np.sum(np.log(gaussian_likelihood(x, means[c], variances[c])))
            
            # Posterior (P(y|x)) = P(y) * P(x|y) = log(P(y)) + log(P(x|y))
            class_probs[c] = prior + likelihood
        
        # Choose the class with the highest posterior
        predictions.append(max(class_probs, key=class_probs.get))
        
        # Log progress every 100 samples
        if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
            logging.info(f"Predicted {idx + 1}/{total_samples} samples... ({(idx + 1) / total_samples * 100:.2f}%)")
    
    prediction_time = time.time() - start_time
    logging.info(f"Prediction completed in {prediction_time:.2f} seconds.\n")
    
    return np.array(predictions)

# Train the model and predict on the test set
y_pred = predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info(f"\n{classification_report(y_test, y_pred)}")

