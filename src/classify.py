import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern

# Load trained model parameters
with open('fruits-360/model_params.pkl', 'rb') as f:
    model_params = pickle.load(f)
