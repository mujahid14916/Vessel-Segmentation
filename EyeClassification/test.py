import keras
from glob import glob
import numpy as np
import os
import shutil

result_dir = 'predicted'
od_dir = os.path.join(result_dir, 'OD')
os_dir = os.path.join(result_dir, 'OS')
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)
if not os.path.isdir(od_dir):
    os.mkdir(od_dir)
if not os.path.isdir(os_dir):
    os.mkdir(os_dir)
model = keras.models.load_model('weights-0495-0.000000.hdf5')
files = glob('../../retcam/*.png')
for file in files:
    image = keras.preprocessing.image.load_img(file, target_size=(250, 250))
    x = model.predict(np.expand_dims(image, axis=0))
    if x[0][0] < 0.5:
        shutil.copy(file, od_dir)
    else:
        shutil.copy(file, os_dir)

