# fruits-360/main.py
from src.preprocess import extract_features
from src.train import train_classifier
from src.classify import classify_image

def main():
    print("Extracting features...")
    extract_features("fruits-360/Train")  # updated path

    print("Training model...")
    train_classifier()  # make sure this is renamed properly in train.py

    print("Classifying new image...")
    image_path = "fruits-360/Test/Apple Red 1/0_100.jpg"  # update as needed
    result = classify_image(image_path)
    print(f"Predicted class: {result}")

if __name__ == "__main__":
    main()
