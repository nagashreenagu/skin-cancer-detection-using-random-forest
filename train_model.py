# train_model.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm
from utils import extract_features, is_blank_image

DATA_DIR = 'data'  # expects data/benign and data/malignant

def gather_dataset(data_dir=DATA_DIR):
    X = []
    y = []
    for label, sub in enumerate(['benign','malignant']):
        folder = os.path.join(data_dir, sub)
        if not os.path.exists(folder):
            continue
        for f in os.listdir(folder):
            p = os.path.join(folder, f)
            if not p.lower().endswith(('.png','.jpg','.jpeg')):
                continue
            if is_blank_image(p):
                print('Skipping blank image', p)
                continue
            try:
                feat = extract_features(p)
                X.append(feat)
                y.append(label)
            except Exception as e:
                print('Error', p, e)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    X, y = gather_dataset()
    if len(X) == 0:
        print('No images found. Please put images into data/benign and data/malignant')
        exit(1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train_s, y_train)

    preds = clf.predict(X_test_s)
    print(classification_report(y_test, preds, target_names=['benign','malignant']))
    print('Confusion matrix:\n', confusion_matrix(y_test, preds))

    joblib.dump(clf, 'models.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print('Saved models.pkl and scaler.pkl')
