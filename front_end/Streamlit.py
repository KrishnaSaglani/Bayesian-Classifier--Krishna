import streamlit as st
# python -m streamlit run front_end/Streamlit.py
import cv2
import numpy as np
import joblib
import pickle
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage.measure import moments_hu
from PIL import Image

RADIUS = 3
N_POINTS = 8 * RADIUS
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0]

def extract_features(img):
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

def predict(image, model_params):
    scaler = model_params['scaler']
    pca = model_params['pca']
    normalizer = model_params['normalizer']

    X = extract_features(image).reshape(1, -1)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    X_norm = normalizer.transform(X_pca)

    class_priors = model_params['class_priors']
    means = model_params['means']
    variances = model_params['variances']
    classes = model_params['unique_classes']

    log_probs = []
    for cls in classes:
        mean = means[cls]
        var = variances[cls]
        prior = class_priors[cls]
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X_norm - mean) ** 2) / var)
        log_probs.append(np.log(prior) + log_likelihood)

    return classes[np.argmax(log_probs)]

# Streamlit UI
st.title("Fruit Image Classifier 🍎🍌🍇\n -Bayesian Decision Theory")
st.write("\nUpload an image of a fruit to classify it.")

model_params = joblib.load("front_end/model.pkl")

uploaded_file = st.file_uploader("Choose a fruit image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    if st.button("Classify"):
        prediction = predict(image_cv, model_params)
        st.success(f"Predicted Fruit: {prediction}")
