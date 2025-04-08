import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
import re
import logging

# Set up logging
logging.basicConfig(filename='extract_test_features.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Normalize similar classes (merge class names like "Apple 1", "Apple small" to "Apple")
def normalize_class(label):
    return re.sub(r'[^a-zA-Z]', '', label)  # Remove numbers and spaces

# Function to extract features from test images
def extract_test_features(test_data_dir):
    features = []
    labels = []

    # Get the total number of images for progress tracking
    total_images = sum([len(os.listdir(os.path.join(test_data_dir, fruit_class))) for fruit_class in os.listdir(test_data_dir) if os.path.isdir(os.path.join(test_data_dir, fruit_class))])
    processed_images = 0

    logging.info(f"Started feature extraction for test data in '{test_data_dir}'...")
    
    for fruit_class in os.listdir(test_data_dir):
        class_path = os.path.join(test_data_dir, fruit_class)
        if not os.path.isdir(class_path):
            continue
        
        # Normalize the class name
        normalized_class = normalize_class(fruit_class)
        logging.info(f"Label before normalization: {fruit_class}, normalized: {normalized_class}")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))  # Resize image to 100x100 pixels
            
            # Extract color histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract texture (LBP)
            lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
            lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()
            
            # Extract shape features (Hu Moments)
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            # Combine features into a feature vector
            feature_vector = np.hstack([hist, lbp_hist, hu_moments])
            
            features.append(feature_vector)
            labels.append(normalized_class)  # Use normalized class as label
            
            processed_images += 1
            if processed_images % 50 == 0:  # Log progress every 50 images
                logging.info(f"Processed {processed_images}/{total_images} images... ({(processed_images/total_images)*100:.2f}%)")
    
    # Save the extracted features and labels to CSV
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv("test_features.csv", index=False)
    logging.info("Feature extraction complete. Saved to test_features.csv")

# Run feature extraction on the test set
extract_test_features("fruits-360/Test")
