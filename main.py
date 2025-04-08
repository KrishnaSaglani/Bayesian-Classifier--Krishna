from src.final_train import train
from src.final_classify import classify
from src.custom_classify import classify_custom_images  
from src.run_pca import run_pca

if __name__ == "__main__":

    print("Starting with pca reduction of extracted features...")

    run_pca(
    train_csv="features/train_advanced_features.csv",
    test_csv="features/test_advanced_features.csv",
    train_out="features/train_adv_pca.csv",
    test_out="features/test_adv_pca.csv",
    custom_csv="features/custom_adv_features.csv",
    custom_out="features/custom_adv_pca.csv"
    )
    print("...................................................................")
    
    print("Starting training process...")
    train()

    print("...................................................................")

    print("\nStarting classification process on test set...")
    classify()

    print("...................................................................")

    print("\nStarting classification on custom images...")
    classify_custom_images()

    print("...................................................................")


