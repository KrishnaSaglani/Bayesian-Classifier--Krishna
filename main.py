# main.py

from src.final_train import train
from src.final_classify import classify

# run src/final_feature_extractor.py once before running this file
# if features folder has no csv files

if __name__ == "__main__":
    print("Starting training process...")
    train()

    print("........................................")

    print("\nStarting classification process...")
    classify()
