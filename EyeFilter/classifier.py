import tensorflow as tf
from glob import glob
import numpy as np
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))

MODEL_WEIGHT_FILE = os.path.join(cur_dir, 'weights-0197-0.0045-0.9951-0.0076-1.0000.hdf5')
# TODO: Need to Fix Model
model = tf.keras.models.load_model(MODEL_WEIGHT_FILE)


def classify(images, min_images=5):
    resized_images = tf.image.resize(images, size=(256, 256)).numpy()
    if np.max(resized_images) > 1:
        resized_images = np.array(resized_images/255., dtype=np.float32)
    predictions = model.predict(resized_images, batch_size=16).flatten()
    print(predictions)
    valid_images = np.zeros(len(images), dtype=np.int)

    order = np.argsort(predictions)[::-1]
    if len(order) > min_images:
        valid_images[order[:min_images]] = 1
        valid_images[predictions >= 0.5] = 1
    else:
        valid_images[:] = 1
    return valid_images

    temp = np.copy(predictions)
    visited = [False] * len(temp)
    for i in range(len(temp)):
        m = -1
        idx = -1
        for j in range(i, len(temp)):
            if temp[j] > m and not visited[j]:
                m = temp[j]
                idx = j
                visited[j] = True
        order.append(idx)
    print(predictions)
    print(order)
    # for i, p in enumerate(predictions):
    #     if p[0] >= 0.5:
    #         valid_images.append(1)
    #     else:
    #         valid_images.append(0)
    return valid_images


def main():
    img = tf.keras.preprocessing.image.load_img(os.path.join(cur_dir, '../../340bff30-20bd-4250-bb36-65b4b57d5c9e_7.png'))
    img = np.expand_dims(np.asarray(img), axis=0)
    print(classify(img))


if __name__ == '__main__':
    main()
