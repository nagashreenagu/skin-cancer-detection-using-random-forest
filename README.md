Skin Cancer Detection — Flask + RandomForest

How to run:
1. Create and activate a Python virtual environment (Python 3.8+).
2. pip install -r requirements.txt
3. (Optional) Use dataset_prep.py to prepare HAM10000 dataset into data/benign and data/malignant
4. Place images into data/benign and data/malignant (a small sample dataset is included).
5. python train_model.py
6. python app.py
7. Open http://127.0.0.1:5000/ in browser. Register, login, upload an image.
