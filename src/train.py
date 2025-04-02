import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import logging
import pickle

# Setup logging (file + console)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[
    logging.FileHandler("train.log"),
    logging.StreamHandler()
])

# Load dataset
data = pd.read_csv('features.csv')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Labels

# Class Merging
class_mapping = {
    "Cabbagered": "Cabbage", "Cabbagewhite": "Cabbage",
    "AppleBraeburn": "Apple", "AppleCore": "Apple", "AppleCrimsonSnow": "Apple",
    "AppleGolden": "Apple", "AppleGrannySmith": "Apple", "ApplePinkLady": "Apple",
    "AppleRed": "Apple", "AppleRedDelicious": "Apple", "AppleRedYellow": "Apple",
    "AppleRotten": "Apple", "Applehit": "Apple", "Appleworm": "Apple",
    "Avocadoripe": "Avocado",
    "BananaLadyFinger": "Banana", "BananaRed": "Banana",
    "Blackberriehalfrippen": "Blackberry", "Blackberrienotrippen": "Blackberry",
    "CherryRainier": "Cherry", "CherryWaxBlack": "Cherry", "CherryWaxRed": "Cherry",
    "CherryWaxYellow": "Cherry", "CherryWaxnotrippen": "Cherry"
}
y = np.array([class_mapping[label] if label in class_mapping else label for label in y])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logging.info("Training started...")

# Class priors P(y)
class_priors = {}
start_time = time.time()

for c in np.unique(y_train):
    class_priors[c] = np.sum(y_train == c) / len(y_train)
    logging.info(f"Class '{c}' prior: {class_priors[c]:.4f}")

logging.info(f"Class priors calculated in {time.time() - start_time:.2f} seconds.\n")

# Means and variances (Per-Class Standardization)
means = {}
variances = {}
start_time = time.time()

for c in np.unique(y_train):
    X_c = X_train[y_train == c]
    means[c] = np.mean(X_c, axis=0)
    variances[c] = np.var(X_c, axis=0) + 1e-4  # **Fix: Smaller Variance Correction**
    
    logging.info(f"Class '{c}' - Mean: {means[c][:3]}... Variance: {variances[c][:3]}...")  # Log first 3 features

logging.info(f"Means & variances computed in {time.time() - start_time:.2f} seconds.\n")

# Gaussian likelihood function
def gaussian_likelihood(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

# Predict function
def predict(X):
    predictions = []
    total_samples = X.shape[0]

    logging.info(f"Making predictions on {total_samples} test samples...")

    start_time = time.time()
    
    for idx, x in enumerate(X):
        class_probs = {}
        for c in np.unique(y_train):
            prior = np.log(class_priors[c] + 1e-9)  # **Fix: Avoid log(0)**
            likelihood = np.sum(np.log(gaussian_likelihood(x, means[c], variances[c]) + 1e-9))  # **Fix: Avoid log(0)**
            class_probs[c] = prior + likelihood  # Log Posterior
        
        predictions.append(max(class_probs, key=class_probs.get))
        
        # Console logs every 100 samples
        if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
            logging.info(f"Predicted {idx + 1}/{total_samples} samples... ({(idx + 1) / total_samples * 100:.2f}%)")
    
    logging.info(f"Prediction completed in {time.time() - start_time:.2f} seconds.\n")
    
    return np.array(predictions)

# Predict and evaluate
y_pred = predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Model Accuracy: {accuracy * 100:.2f}%")
logging.info(f"\n{classification_report(y_test, y_pred)}")

# Save the model parameters to a file for later use in classification
model_params = {
    'class_priors': class_priors,
    'means': means,
    'variances': variances,
    'unique_classes': np.unique(y_train)
}

with open('model_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)

logging.info("Model parameters saved to 'model_params.pkl'")
