# main.py - Main script to run the classifier
from src.preprocess import extract_features
from src.train import train_model
from src.classify import classify_image

def main():
    print("Extracting features...")
    extract_features("Fruits 360/Train")
    
    print("Training model...")
    train_model()
    
    print("Classifying new image...")
    image_path = "Fruits 360/Test/sample_image.jpg"
    result = classify_image(image_path)
    print(f"Predicted class: {result}")

if __name__ == "__main__":
    main()
