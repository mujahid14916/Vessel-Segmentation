import tensorflow as tf
from glob import glob
import numpy as np
import os
import shutil
from patch_generator import square_frame
from tqdm import tqdm

cur_dir = os.path.dirname(os.path.realpath(__file__))
os.sys.path.insert(-1, cur_dir)

MODEL_WEIGHT_FILE = os.path.join(cur_dir, 'EyeClassification/new_model_2/weights-0644-0.0243-0.9909-0.3269-0.9604.hdf5')

model = tf.keras.models.load_model(MODEL_WEIGHT_FILE)


def classify(images):
    resized_images = tf.image.resize(images, size=(256, 256)).numpy()
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


def main():
    files = glob('../neo/*.png')
    res_dir = '../eye_classification_new_neo'
    od_dir = os.path.join(res_dir, 'OD')
    os_dir = os.path.join(res_dir, 'OS')
    if not os.path.isdir(res_dir):
        os.mkdir(res_dir)
    if not os.path.isdir(os_dir):
        os.mkdir(os_dir)
    if not os.path.isdir(od_dir):
        os.mkdir(od_dir)
    images = []
    for file in tqdm(files, desc='Reading Images'):
        image = np.array(tf.keras.preprocessing.image.load_img(file))
        image = square_frame(image)
        images.append(image)
    for file, image in tqdm(zip(files, images), total=len(files)):
        label = classify(np.expand_dims(image, axis=0))
        if label[0] == 'OD':
            shutil.copy(file, od_dir)
        else:
            shutil.copy(file, os_dir)


if __name__ == '__main__':
    main()


