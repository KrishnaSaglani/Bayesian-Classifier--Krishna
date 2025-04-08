import os
import cv2
import numpy as np
import pandas as pd
import logging
from skimage.feature import local_binary_pattern
import re

def extract_features(data_dir, output_csv, log_file):
    # Set up logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')
    
    features = []
    labels = []
    
    # Get total number of images for progress tracking
    total_images = sum([
        len(os.listdir(os.path.join(data_dir, fruit_class))) 
        for fruit_class in os.listdir(data_dir) 
        if os.path.isdir(os.path.join(data_dir, fruit_class))
    ])
    processed_images = 0
    
    logging.info(f"Started feature extraction from {data_dir}. Total images to process: {total_images}")
    print(f"Started feature extraction from {data_dir}. Total images to process: {total_images}")
    
    for fruit_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, fruit_class)
        if not os.path.isdir(class_path):
            continue
        
        # Extract base fruit name (e.g., 'Apple' from 'Apple1', 'Apple small')
        base_class = re.sub(r'[^a-zA-Z]', '', fruit_class)
        logging.info(f"Processing class: {base_class}")
        print(f"Processing class: {base_class}")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Skipping unreadable image: {img_name}")
                continue
            
            img = cv2.resize(img, (100, 100))
            
            # Color histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Grayscale conversion
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Texture (LBP)
            lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
            lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()
            
            # Shape (Hu Moments)
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            # Combine features
            feature_vector = np.hstack([hist, lbp_hist, hu_moments])
            features.append(feature_vector)
            labels.append(base_class)
            
            processed_images += 1
            if processed_images % 50 == 0:
                percent = (processed_images / total_images) * 100
                logging.info(f"Processed {processed_images}/{total_images} images ({percent:.2f}%)")
                print(f"Processed {processed_images}/{total_images} images ({percent:.2f}%)")
    
    # Save to CSV
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    
    logging.info(f"Feature extraction complete. Saved to {output_csv}")
    print(f"Feature extraction complete. Saved to {output_csv}")

# Only run when called directly
if __name__ == "__main__":
    # Training set
    extract_features(
        data_dir="fruits-360/Training",
        output_csv="train_features.csv",
        log_file="extract_train_features.log"
    )
    
    # Test set
    extract_features(
        data_dir="fruits-360/Test",
        output_csv="test_features.csv",
        log_file="extract_test_features.log"
    )
