import keras
from glob import glob
import numpy as np
import os
import shutil
import pandas as pd
from helper_utils import is_fundus

IMAGES_DIRECTORIES = ['../../neo/*.png', '../../retcam/*.png']
MODEL_WEIGHT_FILE = 'weights-1859-0.0198-0.9883-0.0826-0.9400.hdf5'

result_dir = 'predicted'
od_dir = os.path.join(result_dir, 'OD')
os_dir = os.path.join(result_dir, 'OS')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if not os.path.isdir(od_dir):
    os.mkdir(od_dir)
if not os.path.isdir(os_dir):
    os.mkdir(os_dir)
model = keras.models.load_model(MODEL_WEIGHT_FILE)
files = []
for img_dir in IMAGES_DIRECTORIES:
    files.extend(glob(img_dir))
label = None
data_source = 'neoDataSet1/imageData/'
df = pd.read_csv('neoDataSet1/metadata/set1EyeLabels.csv', header=None, names=['image_name', 'label'], index_col=0)
for file in files:
    file_name = file.replace('\\', '/').split('/')[-1]
    print(file)
    image = keras.preprocessing.image.load_img(file, target_size=(250, 250))
    if is_fundus(image):
        x = model.predict(np.expand_dims(image, axis=0))
        if x[0][0] < 0.5:
            label = 'OD'
            shutil.copy(file, od_dir)
        else:
            label = 'OS'
            shutil.copy(file, os_dir)
        shutil.copy(file, data_source)
        if file_name not in df.index:
            df = df.append(pd.Series(data=[label], index=['label'], name=file_name))
    else:
        print("Skipped:", file)
df.to_csv('neoDataSet1/metadata/labels.csv', header=False)
