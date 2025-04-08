import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer

def run_pca(
    train_csv,
    test_csv,
    train_out,
    test_out,
    custom_csv=None,
    custom_out=None
):
    """
    Applies standardization, PCA, and normalization to the dataset.
    Optionally processes custom image features if provided.

    Parameters:
    - train_csv (str): Path to training features CSV with 'label' column
    - test_csv (str): Path to testing features CSV with 'label' column
    - train_out (str): Path to save processed training output
    - test_out (str): Path to save processed testing output
    - custom_csv (str, optional): Path to custom image features CSV
    - custom_out (str, optional): Path to save processed custom output
    """

    print("[1/6] Loading data...")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    print(f"Loaded training data: {train_df.shape}, testing data: {test_df.shape}")

    print("[2/6] Separating labels from features...")
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]
    print("Separated labels.")

    print("[3/6] Standardizing features (zero mean, unit variance)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Standardization complete.")

    print("[4/6] Applying PCA to retain 95% variance...")
    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    print(f"PCA reduced dimensions from {X_train.shape[1]} to {X_train_pca.shape[1]}.")

    print("[5/6] Normalizing feature vectors (L2 norm)...")
    normalizer = Normalizer()
    X_train_pca_norm = normalizer.fit_transform(X_train_pca)
    X_test_pca_norm = normalizer.transform(X_test_pca)
    print("Normalization complete.")

    print("[6/6] Reconstructing and saving final datasets...")
    train_pca_df = pd.DataFrame(X_train_pca_norm)
    train_pca_df["label"] = y_train.values
    test_pca_df = pd.DataFrame(X_test_pca_norm)
    test_pca_df["label"] = y_test.values

    train_pca_df.to_csv(train_out, index=False)
    test_pca_df.to_csv(test_out, index=False)

    print(" PCA + Normalization pipeline complete.")
    print(f"Train shape: {train_pca_df.shape}, Test shape: {test_pca_df.shape}")

    # Optional: Handle custom image features
    if custom_csv and custom_out and os.path.exists(custom_csv):
        print("\n Detected custom image feature file. Processing...")
        custom_df = pd.read_csv(custom_csv)
        if "label" not in custom_df.columns:
            print(" 'label' column not found in custom CSV. Skipping.")
        else:
            X_custom = custom_df.drop(columns=["label"])
            y_custom = custom_df["label"]

            X_custom_scaled = scaler.transform(X_custom)
            X_custom_pca = pca.transform(X_custom_scaled)
            X_custom_pca_norm = normalizer.transform(X_custom_pca)

            custom_pca_df = pd.DataFrame(X_custom_pca_norm)
            custom_pca_df["label"] = y_custom.values
            custom_pca_df.to_csv(custom_out, index=False)

            print(f" Custom PCA completed. Saved to {custom_out}")

