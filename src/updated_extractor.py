import os
import cv2
import numpy as np
import pandas as pd
import logging
from skimage.feature import local_binary_pattern

def extract_features(data_dir, output_csv, log_file):
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

    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

    features = []
    labels = []

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

        base_class = class_mapping.get(fruit_class, None)
        if base_class is None:
            logging.warning(f"No mapping found for: {fruit_class}. Skipping.")
            print(f"Skipping class (no mapping): {fruit_class}")
            continue

        logging.info(f"Processing class: {fruit_class} -> {base_class}")
        print(f"Processing class: {fruit_class} -> {base_class}")

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                logging.warning(f"Skipping unreadable image: {img_name}")
                continue

            img = cv2.resize(img, (100, 100))

            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            lbp = local_binary_pattern(gray, P=24, R=3, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
            lbp_hist = lbp_hist.astype("float") / lbp_hist.sum()

            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            feature_vector = np.hstack([hist, lbp_hist, hu_moments])
            features.append(feature_vector)
            labels.append(base_class)

            processed_images += 1
            if processed_images % 50 == 0:
                percent = (processed_images / total_images) * 100
                logging.info(f"Processed {processed_images}/{total_images} images ({percent:.2f}%)")
                print(f"Processed {processed_images}/{total_images} images ({percent:.2f}%)")

    df = pd.DataFrame(features)
    df['label'] = labels
    df.to_csv(output_csv, index=False)

    logging.info(f"Feature extraction complete. Saved to {output_csv}")
    print(f"Feature extraction complete. Saved to {output_csv}")


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
