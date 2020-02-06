import pandas as pd
import numpy as np
from glob import glob
import os
import cv2
from helper_utils import resize_image

images_path = glob('neoDataSet1/imageData/*.png')

df = pd.read_csv('neoDataSet1/metadata/corrected_labels.csv', header=None, names=['image_name', 'label'], index_col=0)

unique_labels = pd.unique(df['label'])

DATASET_ROOT_DIR = 'dataset'
if not os.path.isdir(DATASET_ROOT_DIR):
    os.mkdir(DATASET_ROOT_DIR)
for label in unique_labels:
    label_dir = os.path.join(DATASET_ROOT_DIR, label)
    if not os.path.isdir(label_dir):
        os.mkdir(label_dir)


for image_path in images_path:
    image_name = image_path.replace('\\', '/').split('/')[-1]
    print(image_name)
    if image_name in df.index:
        image = cv2.imread(image_path)
        image = resize_image(image, target_shape=(500, 500))
        label = df.loc[image_name].label
        dest_dir = os.path.join(DATASET_ROOT_DIR, label)
        cv2.imwrite(os.path.join(dest_dir, image_name), image)
