import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern

# Load trained model parameters
with open('fruits-360/model_params.pkl', 'rb') as f:
    model_params = pickle.load(f)

class_priors = model_params['class_priors']
means = model_params['means']
variances = model_params['variances']
unique_classes = model_params['unique_classes']


def gaussian_likelihood(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

def classify_image(image_path):

    for c in unique_classes:
        prior = np.log(class_priors[c] + 1e-9)
        likelihood = np.sum(np.log(gaussian_likelihood(x, means[c], variances[c]) + 1e-9))
        class_probs[c] = prior + likelihood  # Log posterior

    return max(class_probs, key=class_probs.get)
