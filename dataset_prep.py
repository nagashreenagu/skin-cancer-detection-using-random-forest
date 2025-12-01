# dataset_prep.py
# Helper to prepare HAM10000-style CSV and images into data/benign and data/malignant folders.
# Usage:
# python dataset_prep.py metadata.csv images_dir

import os
import sys
import shutil
import csv

def map_label(row):
    # Example mapping: malignant classes in HAM10000 include 'mel' (melanoma).
    # Adjust mapping as needed for your CSV column names and labels.
    lesion = row.get('dx') or row.get('label') or row.get('diagnosis') or ''
    lesion = lesion.lower()
    malignant_keywords = set(['mel', 'melanoma', 'bcc', 'scc'])
    for k in malignant_keywords:
        if k in lesion:
            return 'malignant'
    return 'benign'

def main(csv_path, images_dir):
    if not os.path.exists(csv_path):
        print('CSV not found', csv_path); return
    if not os.path.exists(images_dir):
        print('Images dir not found', images_dir); return
    out_base = 'data'
    benign_dir = os.path.join(out_base, 'benign')
    mal_dir = os.path.join(out_base, 'malignant')
    os.makedirs(benign_dir, exist_ok=True)
    os.makedirs(mal_dir, exist_ok=True)

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row.get('image_id') or row.get('image') or row.get('filename') or row.get('name')
            if not image_id:
                continue
            # try common extensions
            found = None
            for ext in ['.jpg','.png','.jpeg','.tif']:
                p = os.path.join(images_dir, image_id + ext)
                if os.path.exists(p):
                    found = p; break
            if not found:
                # maybe filename column has extension
                p = os.path.join(images_dir, image_id)
                if os.path.exists(p):
                    found = p
            if not found:
                continue
            cls = map_label(row)
            if cls == 'malignant':
                shutil.copy(found, os.path.join(mal_dir, os.path.basename(found)))
            else:
                shutil.copy(found, os.path.join(benign_dir, os.path.basename(found)))
    print('Done. Data prepared under data/benign and data/malignant')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python dataset_prep.py metadata.csv images_dir')
    else:
        main(sys.argv[1], sys.argv[2])
