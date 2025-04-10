import os
import numpy as np
import cv2
import joblib
import pickle
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import moments_hu
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.decomposition import PCA

# Some headers to make my work easier later
TRAIN_DIR = 'fruits-360/Training'
RADIUS = 3
N_POINTS = 8 * RADIUS
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]

# Class Mapping Dictionary
class_mapping = {
    "Apple 10": "Apple", "Apple 12": "Apple", "Apple 13": "Apple", "Apple 14": "Apple", "Apple 17": "Apple",
    "Apple 19": "Apple", "Apple 6": "Apple", "Apple 9": "Apple", "Apple Braeburn 1": "Apple",
    "Apple Core 1": "Apple", "Apple Crimson Snow 1": "Apple", "Apple Golden 1": "Apple",
    "Apple Golden 2": "Apple", "Apple Golden 3": "Apple", "Apple Granny Smith 1": "Apple",
    "Apple hit 1": "Apple", "Apple Pink Lady 1": "Apple", "Apple Red 1": "Apple", "Apple Red 2": "Apple",
    "Apple Red 3": "Apple", "Apple Red Delicious 1": "Apple", "Apple Red Yellow 1": "Apple",
    "Apple Red Yellow 2": "Apple", "Apple Rotten 1": "Apple", "Apple worm 1": "Apple",
    "Apricot 1": "Apricot", "Avocado 1": "Avocado", "Avocado ripe 1": "Avocado",
    "Banana 1": "Banana", "Banana 3": "Banana", "Banana Lady Finger 1": "Banana", "Banana Red 1": "Banana",
    "Beans 1": "Beans", "Beetroot 1": "Beetroot", "Blackberrie 1": "Blackberry", "Blackberrie 2": "Blackberry",
    "Blackberrie half rippen 1": "Blackberry", "Blackberrie not rippen 1": "Blackberry", "Blueberry 1": "Blueberry",
    "Cabbage red 1": "Cabbage", "Cabbage white 1": "Cabbage", "Cactus fruit 1": "Cactus fruit",
    "Cactus fruit green 1": "Cactus fruit", "Cactus fruit red 1": "Cactus fruit", "Cantaloupe 1": "Cantaloupe",
    "Cantaloupe 2": "Cantaloupe", "Carambula 1": "Carambula", "Carrot 1": "Carrot", "Cauliflower 1": "Cauliflower",
    "Cherimoya 1": "Cherimoya", "Cherry 1": "Cherry", "Cherry 2": "Cherry", "Cherry Rainier 1": "Cherry",
    "Cherry Wax Black 1": "Cherry", "Cherry Wax not rippen 1": "Cherry", "Cherry Wax Red 1": "Cherry",
    "Cherry Wax Yellow 1": "Cherry", "Chestnut 1": "Chestnut", "Clementine 1": "Clementine",
    "Cocos 1": "Coconut", "Corn 1": "Corn", "Corn Husk 1": "Corn", "Cucumber 1": "Cucumber",
    "Cucumber 10": "Cucumber", "Cucumber 3": "Cucumber", "Cucumber 9": "Cucumber",
    "Cucumber Ripe 1": "Cucumber", "Cucumber Ripe 2": "Cucumber", "Dates 1": "Dates",
    "Eggplant 1": "Eggplant", "Eggplant long 1": "Eggplant", "Fig 1": "Fig", "Ginger Root 1": "Ginger",
    "Gooseberry 1": "Gooseberry", "Granadilla 1": "Granadilla", "Grape Blue 1": "Grape",
    "Grape Pink 1": "Grape", "Grape White 1": "Grape", "Grape White 2": "Grape", "Grape White 3": "Grape",
    "Grape White 4": "Grape", "Grapefruit Pink 1": "Grapefruit", "Grapefruit White 1": "Grapefruit",
    "Guava 1": "Guava", "Hazelnut 1": "Hazelnut", "Huckleberry 1": "Huckleberry", "Kaki 1": "Kaki",
    "Kiwi 1": "Kiwi", "Kohlrabi 1": "Kohlrabi", "Kumquats 1": "Kumquat", "Lemon 1": "Lemon",
    "Lemon Meyer 1": "Lemon", "Limes 1": "Lime", "Lychee 1": "Lychee", "Mandarine 1": "Mandarine",
    "Mango 1": "Mango", "Mango Red 1": "Mango", "Mangostan 1": "Mangosteen", "Maracuja 1": "Passion Fruit",
    "Melon Piel de Sapo 1": "Melon", "Mulberry 1": "Mulberry", "Nectarine 1": "Nectarine",
    "Nectarine Flat 1": "Nectarine", "Nut Forest 1": "Nut", "Nut Pecan 1": "Nut", "Onion Red 1": "Onion",
    "Onion Red Peeled 1": "Onion", "Onion White 1": "Onion", "Orange 1": "Orange", "Papaya 1": "Papaya",
    "Passion Fruit 1": "Passion Fruit", "Peach 1": "Peach", "Peach 2": "Peach", "Peach Flat 1": "Peach",
    "Pear 1": "Pear", "Pear 2": "Pear", "Pear 3": "Pear", "Pear Abate 1": "Pear", "Pear Forelle 1": "Pear",
    "Pear Kaiser 1": "Pear", "Pear Monster 1": "Pear", "Pear Red 1": "Pear", "Pear Stone 1": "Pear",
    "Pear Williams 1": "Pear", "Pepino 1": "Pepino", "Pepper Green 1": "Pepper", "Pepper Orange 1": "Pepper",
    "Pepper Red 1": "Pepper", "Pepper Yellow 1": "Pepper", "Physalis 1": "Physalis",
    "Physalis with Husk 1": "Physalis", "Pineapple 1": "Pineapple", "Pineapple Mini 1": "Pineapple",
    "Pistachio 1": "Pistachio", "Pitahaya Red 1": "Pitahaya", "Plum 1": "Plum", "Plum 2": "Plum",
    "Plum 3": "Plum", "Pomegranate 1": "Pomegranate", "Pomelo Sweetie 1": "Pomelo",
    "Potato Red 1": "Potato", "Potato Red Washed 1": "Potato", "Potato Sweet 1": "Potato",
    "Potato White 1": "Potato", "Quince 1": "Quince", "Quince 2": "Quince", "Quince 3": "Quince",
    "Quince 4": "Quince", "Rambutan 1": "Rambutan", "Raspberry 1": "Raspberry", "Redcurrant 1": "Redcurrant",
    "Salak 1": "Salak", "Strawberry 1": "Strawberry", "Strawberry Wedge 1": "Strawberry", "Tamarillo 1": "Tamarillo",
    "Tangelo 1": "Tangelo", "Tomato 1": "Tomato", "Tomato 2": "Tomato", "Tomato 3": "Tomato",
    "Tomato 4": "Tomato", "Tomato Cherry Red 1": "Tomato", "Tomato Heart 1": "Tomato",
    "Tomato Maroon 1": "Tomato", "Tomato not Ripened 1": "Tomato", "Tomato Yellow 1": "Tomato",
    "Walnut 1": "Walnut", "Watermelon 1": "Watermelon", "Zucchini 1": "Zucchini",
    "Zucchini dark 1": "Zucchini"
}


# Feature extractor
def extract_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (100, 100))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256])
    hsv_hist = cv2.normalize(hsv_hist, hsv_hist).flatten()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    haralick_features = np.array([contrast, correlation, energy, homogeneity])

    moments = cv2.moments(gray)
    hu_moments = cv2.HuMoments(moments).flatten()
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    feature_vector = np.hstack([hsv_hist, lbp_hist, haralick_features, hu_moments])
    return feature_vector

# Extract features
X = []
y = []

print("Starting feature extraction...")
processed_images = 0

total_images = sum([
            len(os.listdir(os.path.join(TRAIN_DIR, fruit_class)))
            for fruit_class in os.listdir(TRAIN_DIR)
            if os.path.isdir(os.path.join(TRAIN_DIR, fruit_class))
        ])


for folder in os.listdir(TRAIN_DIR):
    folder_path = os.path.join(TRAIN_DIR, folder)
    if os.path.isdir(folder_path):
        base_class = class_mapping.get(folder, folder)
        if base_class is None:
                print(f"Skipping class (no mapping): {folder}")
                continue
        print(f"Processing class: {folder} -> {base_class}")
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            features = extract_features(file_path)
            X.append(features)
            y.append(base_class)  # Use class_mapping if available

            processed_images += 1
            if processed_images % 50 == 0:
                percent = (processed_images / total_images) * 100
                print(f"Processed {processed_images}/{total_images} images ({percent:.2f}%)")

X = np.array(X)
y = np.array(y)

print(f"Total samples extracted: {len(X)}")
print("Starting preprocessing...")

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Standard scaling completed.")

pca = PCA(n_components=100)
X_pca = pca.fit_transform(X_scaled)
print("PCA transformation completed.")

normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X_pca)
print("Normalization completed.")

# Train Gaussian Naive Bayes manually
unique_classes = np.unique(y)
class_priors = {}
means = {}
variances = {}

print("Starting model training...")

for cls in unique_classes:
    X_cls = X_normalized[y == cls]
    class_priors[cls] = X_cls.shape[0] / X_normalized.shape[0]
    means[cls] = np.mean(X_cls, axis=0)
    variances[cls] = np.var(X_cls, axis=0) + 1e-6

print(f"Trained on {len(unique_classes)} unique classes.")

model_params = {
    'class_priors': class_priors,
    'means': means,
    'variances': variances,
    'unique_classes': unique_classes,
    'pca_components': pca.components_,
    'scaler': scaler,
    'pca': pca,
    'normalizer': normalizer
}

os.makedirs('model', exist_ok=True)
with open('front_end/model.pkl', 'wb') as f:
    pickle.dump(model_params, f)

print(" Model parameters saved successfully to 'front_end/model.pkl'.")
