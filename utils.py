# utils.py
import os
import numpy as np
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray

def load_image(path, size=(128,128)):
    with Image.open(path) as im:
        im = im.convert('RGB')
        im = im.resize(size)
        arr = np.array(im)
    return arr

def color_histogram(img, bins=32):
    chans = []
    for i in range(3):
        hist,_ = np.histogram(img[:,:,i], bins=bins, range=(0,255), density=True)
        chans.append(hist)
    return np.concatenate(chans)

def hog_features(img_gray, pixels_per_cell=(16,16)):
    fd = hog(img_gray, pixels_per_cell=pixels_per_cell, cells_per_block=(2,2), feature_vector=True)
    return fd

def extract_features(path, size=(128,128)):
    img = load_image(path, size=size)
    hist = color_histogram(img, bins=32)
    gray = rgb2gray(img)
    hogf = hog_features(gray, pixels_per_cell=(16,16))
    feat = np.concatenate([hist, hogf])
    return feat

def is_blank_image(path, threshold=1e-3):
    arr = load_image(path, size=(64,64))
    return arr.std() < threshold
