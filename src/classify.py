import pandas as pd
import numpy as np
import pickle
import time
import re

# Load the trained model parameters
with open('model_params.pkl', 'rb') as f:
    model_data = pickle.load(f)
    class_priors = model_data['class_priors']
    means = model_data['means']
    variances = model_data['variances']
    unique_classes = model_data['unique_classes']

def normalize_class(label):
    return re.sub(r'[^a-zA-Z]', '', label)  # Remove numbers and spaces

# normalized_label = normalize_class(fruit_class)
# logging.info(f"Label before normalization: {fruit_class}, normalized: {normalized_label}")

def gaussian_likelihood(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x - mean) ** 2 / (2 * var))

def predict(X):
    predictions = []
    total_samples = X.shape[0]
    
    print(f"Classifying {total_samples} new samples...")
    start_time = time.time()
    
    for idx, x in enumerate(X):
        class_probs = {}
        for c in unique_classes:
            prior = np.log(class_priors[c])
            likelihood = np.sum(np.log(gaussian_likelihood(x, means[c], variances[c])))
            class_probs[c] = prior + likelihood
        
        predictions.append(max(class_probs, key=class_probs.get))
        
        if (idx + 1) % 100 == 0 or (idx + 1) == total_samples:
            print(f"Classified {idx + 1}/{total_samples} samples... ({(idx + 1) / total_samples * 100:.2f}%)")
    
    classification_time = time.time() - start_time
    print(f"Classification completed in {classification_time:.2f} seconds.")
    
    return np.array(predictions)

# Load test data
new_data = pd.read_csv('test_samples.csv')
X_new = new_data.values  # Extract feature values

# Predict labels for new test samples
y_pred = predict(X_new)

# Save predictions to a file
output_df = pd.DataFrame({'Predicted Class': y_pred})
output_df.to_csv('predictions.csv', index=False)
print("Predictions saved to predictions.csv")
