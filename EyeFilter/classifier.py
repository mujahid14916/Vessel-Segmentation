import tensorflow as tf
from glob import glob
import numpy as np
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))

MODEL_WEIGHT_FILE = os.path.join(cur_dir, 'weights-0150-0.2902-0.9153-0.3847-0.9505.hdf5')

model = tf.keras.models.load_model(MODEL_WEIGHT_FILE)


def classify(images):
    resized_images = tf.image.resize(images, size=(256, 256)).numpy()
    if np.max(resized_images) > 1:
        resized_images = np.array(resized_images/255., dtype=np.float32)
    predictions = model.predict(resized_images, batch_size=16)
    valid_images = []
    for i, p in enumerate(predictions):
        if p[0] >= 0.5:
            valid_images.append(images[i])
    return np.array(valid_images)



