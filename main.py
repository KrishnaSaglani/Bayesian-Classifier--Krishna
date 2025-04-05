# main.py

from src.final_train import train
from src.final_classify import classify

if __name__ == "__main__":
    print("Starting training process...")
    train()

    print("\nStarting classification process...")
    classify()
