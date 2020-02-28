import tensorflow as tf
from glob import glob
import numpy as np
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))

MODEL_WEIGHT_FILE = os.path.join(cur_dir, 'weights-0069-0.3052-0.9114-0.2536-0.9451.hdf5')

model = tf.keras.models.load_model(MODEL_WEIGHT_FILE)


def classify(images, min_images=5):
    resized_images = tf.image.resize(images, size=(256, 256)).numpy()
    if np.max(resized_images) > 1:
        resized_images = np.array(resized_images/255., dtype=np.float32)
    predictions = model.predict(resized_images, batch_size=16)
    valid_images = []

    order = []
    temp = np.copy(predictions)
    visited = [False] * len(temp)
    for i in range(len(temp)):
        m = -1
        idx = -1
        for j in range(1, len(temp) - i):
            if m > temp[j] and not visited:
                m = temp[j]
                idx = j
        order.append(idx)

    for i, p in enumerate(predictions):
        if p[0] >= 0.5:
            valid_images.append(1)
        else:
            valid_images.append(0)
    return valid_images



