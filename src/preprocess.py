import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
import re

def extract_features(data_dir):
    features = []
    labels = []
    
    # Get the total number of images for progress tracking
    total_images = sum([len(os.listdir(os.path.join(data_dir, fruit_class))) for fruit_class in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, fruit_class))])
    processed_images = 0
    
    print(f"Started feature extraction from {data_dir}. Total images to process: {total_images}")
    
    for fruit_class in os.listdir(data_dir):
        class_path = os.path.join(data_dir, fruit_class)
        if not os.path.isdir(class_path):
            continue
        
        # Extract base fruit name (e.g., 'Apple' from 'Apple1', 'Apple small')
        base_class = re.sub(r'[^a-zA-Z]', '', fruit_class)  # Remove numbers and spaces
        print(f"Processing class: {base_class}")
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Skipping {img_name} due to read error.")
                continue
            
            img = cv2.resize(img, (100, 100))
            
            # Extract color histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Extract texture from local binary pattern function
            lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
            lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()
            
            # Extract shape features (Hu Moments)
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
            
            # Combine features
            feature_vector = np.hstack([hist, lbp_hist, hu_moments])
            
            features.append(feature_vector)
            labels.append(base_class)  # Store the base fruit name
            
            # Track progress
            processed_images += 1
            if processed_images % 50 == 0:  # Log progress every 50 images
                print(f"Processed {processed_images}/{total_images} images... ({(processed_images/total_images)*100:.2f}%)")
    
    # Save to CSV
    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv("train_features.csv", index=False)
    print("Feature extraction complete. Saved to features.csv")

# Run training feature extraction
extract_features("fruits-360/Training")
