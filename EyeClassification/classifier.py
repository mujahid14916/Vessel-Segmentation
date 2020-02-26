import tensorflow as tf
from glob import glob
import numpy as np
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.insert(-1, cur_dir)

from helper_utils import is_fundus
MODEL_WEIGHT_FILE = os.path.join(cur_dir, 'weights-1859-0.0198-0.9883-0.0826-0.9400.hdf5')

model = tf.keras.models.load_model(MODEL_WEIGHT_FILE)


def classify(images):
    resized_images = tf.image.resize(images, size=(250, 250)).numpy()
    if np.max(resized_images) > 1:
        resized_images = np.array(resized_images/255., dtype=np.float32)
    # valid_index = []
    # invalid_index = []
    # for i, image in enumerate(resized_images):
    #     if is_fundus(image):
    #         valid_index.append(i)
    #     else:
    #         invalid_index.append(i)
    # resized_images = resized_images[valid_index]
    prediction = model.predict(resized_images, batch_size=16)
    classes =[]
    for p in prediction:
        if p[0] < 0.5:
            label = 'OD'
        else:
            label = 'OS'
        classes.append(label)
    return classes


