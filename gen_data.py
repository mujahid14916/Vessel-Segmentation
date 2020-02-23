from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm
from keras.preprocessing.image import *


PATCH_SIZE = (256, 256)       # (height, width)
TOTAL_PATCHES = 500
# IMG_MAX_HEIGHT = 800
# IMG_MIN_HEIGHT = 500
RESULT_DIR = 'training_dataset/patches'
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)


def add_sp_noise(img):  # Salt Pepper noise
    max_val = 255
    prob = np.random.randint(5, 150) / 1000
    randn = np.random.randint(-int(max_val*prob), int(max_val*prob), size=(img.shape))
    if np.max(img) <= 1:
        randn = randn / 255
        max_val = 1
    return np.clip(img + randn, 0, max_val)


def data_generator(dataset_root_dir, image_dir, label_dir, image_ext, batch_size, patch_size=(64, 64), image_min_max_hgt=(300, 600), caps=False):
    k = 0
    files = glob('{}/{}/*.{}'.format(dataset_root_dir, image_dir, image_ext))
    images = []
    labels = []
    for file in files:
        img = np.asarray(Image.open(file))
        lbl = np.asarray(Image.open(file.replace(image_dir, label_dir)))
        if np.max(img) > 1:
            data_img = np.array(img / 255, dtype=np.float32)
        if np.max(lbl) > 1:
            data_lbl = np.array(lbl / 255, dtype=np.float32)
        if len(data_img.shape) == 2:
            data_img = np.expand_dims(data_img, axis=-1)
        if len(data_lbl.shape) == 2:
            data_lbl = np.expand_dims(data_lbl, axis=-1)
        images.append(data_img)
        labels.append(data_lbl)

        # Append Scaled Images
        # for _ in range(5):
        #     img_height = np.random.randint(low=image_min_max_hgt[0], high=image_min_max_hgt[1]+1)
        #     img_width = int(img.shape[1] / img.shape[0] * img_height)   # original_width / original_height * height
        #     data_img = cv2.resize(img, (img_width, img_height))     # OpenCV (width, height) format
        #     data_lbl = cv2.resize(lbl, (img_width, img_height))
        #     if np.max(data_img) > 1:
        #         data_img = np.array(data_img / 255, dtype=np.float32)
        #     if np.max(data_lbl) > 1:
        #         data_lbl = np.array(data_lbl / 255, dtype=np.float32)
        #     images.append(data_img)
        #     labels.append(data_lbl)

    while True:
        X = []
        Y = []
        b = 0
        while b < batch_size:
            index = np.random.randint(len(images))
            image = images[index]
            img_shp = image.shape
            label = labels[index]
            if img_shp[0] <= patch_size[0] or img_shp[1] <= patch_size[1]:
                continue
            rnd_height = np.random.randint(img_shp[0] - patch_size[0])
            rnd_width = np.random.randint(img_shp[1] - patch_size[1])
            patch_img = image[rnd_height:rnd_height+patch_size[0], rnd_width:rnd_width+patch_size[1], ...]
            patch_lbl = label[rnd_height:rnd_height+patch_size[0], rnd_width:rnd_width+patch_size[1], ...]
            img_and_mask = np.concatenate((patch_img, patch_lbl), axis=2)
            # Horizontal Flipping
            if np.random.randint(0, 100) == 0:
                img_and_mask = img_and_mask[:, ::-1, ...]
            # Vertical Flipping
            if np.random.randint(0, 100) == 0:
                img_and_mask = img_and_mask[::-1, :, ...]
            # # Patch Stretching
            # if np.random.randint(0, 100) == 0:
            #     height_scale = 1 + np.random.rand()
            #     width_scale = 1 + np.random.rand()
            #     # Horizontal
            #     if np.random.choice([True, False]):
            #         patch_img = cv2.resize(patch_img, (int(patch_size[1] * width_scale), patch_size[0]))
            #         patch_lbl = cv2.resize(patch_lbl, (int(patch_size[1] * width_scale), patch_size[0]))
            #     # Vertical
            #     if np.random.choice([True, False]):
            #         patch_img = cv2.resize(patch_img, (patch_size[1], int(patch_size[0] * height_scale)))
            #         patch_lbl = cv2.resize(patch_lbl, (patch_size[1], int(patch_size[0] * height_scale)))
            #     if len(patch_img.shape) < 3:
            #         patch_img = np.expand_dims(patch_img, axis=-1)
            #         patch_lbl = np.expand_dims(patch_lbl, axis=-1)
            #     img_and_mask[:, :, :1] = patch_img[:patch_size[0], :patch_size[1], ...]
            #     img_and_mask[:, :, 1:] = patch_lbl[:patch_size[0], :patch_size[1], ...]

            if np.random.randint(0, 100) == 0:
                img_and_mask = random_zoom(
                    img_and_mask, zoom_range=(0.7, 1.3),
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0
                )
            if np.random.randint(0, 1000) == 0:
                img_and_mask = random_shift(        # very low prob
                    img_and_mask, wrg=1.5, hrg=1.5,
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0,
                )
            if np.random.randint(0, 100) == 0:
                img_and_mask = random_rotation(
                    img_and_mask, rg=45,
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0
                )
            if np.random.randint(0, 100) == 0:
                img_and_mask[:, :, :1] = random_brightness(
                    patch_img,
                    brightness_range=(0.8, 1.1)
                )/255
            if np.random.randint(0, 100) == 0:
                img_and_mask = random_shear(
                    img_and_mask, intensity = 45,
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0,
                )

            patch_img = img_and_mask[:, :, :1]
            patch_lbl = img_and_mask[:, :, 1:]
            if np.random.randint(0, 100) == 0:
                patch_img = add_sp_noise(patch_img)
            # Can be ignored
            if np.sum(patch_lbl) == 0 and np.random.randint(0, 100) > 0:    # 1% chance of selecting all negative sample
                continue
            # X.append(np.expand_dims(patch_img, axis=-1))
            # Y.append(np.expand_dims(patch_lbl, axis=-1))
            X.append(patch_img)
            Y.append(patch_lbl)
            # k += 1
            b += 1
        if caps:
            x, y = np.array(X), np.array(Y)
            yield ([x, y], [y, y*x])
        else:

            yield np.array(X), np.array(Y)


def full_image_generator(dataset_root_dir, image_dir, label_dir, image_ext, batch_size, image_size=(256, 256), caps=False):
    k = 0
    files = glob('{}/{}/*.{}'.format(dataset_root_dir, image_dir, image_ext))
    images = []
    labels = []
    for file in files:
        img = np.asarray(Image.open(file))
        lbl = np.asarray(Image.open(file.replace(image_dir, label_dir)))
        img = cv2.resize(img, dsize=(image_size[1], image_size[0]))
        lbl = cv2.resize(lbl, dsize=(image_size[1], image_size[0]))
        if np.max(img) > 1:
            data_img = np.array(img / 255, dtype=np.float32)
        if np.max(lbl) > 1:
            data_lbl = np.array(lbl / 255, dtype=np.float32)
        if len(data_img.shape) == 2:
            data_img = np.expand_dims(data_img, axis=-1)
        if len(data_lbl.shape) == 2:
            data_lbl = np.expand_dims(data_lbl, axis=-1)
        images.append(data_img)
        labels.append(data_lbl)

        # Append Scaled Images
        # for _ in range(5):
        #     img_height = np.random.randint(low=image_min_max_hgt[0], high=image_min_max_hgt[1]+1)
        #     img_width = int(img.shape[1] / img.shape[0] * img_height)   # original_width / original_height * height
        #     data_img = cv2.resize(img, (img_width, img_height))     # OpenCV (width, height) format
        #     data_lbl = cv2.resize(lbl, (img_width, img_height))
        #     if np.max(data_img) > 1:
        #         data_img = np.array(data_img / 255, dtype=np.float32)
        #     if np.max(data_lbl) > 1:
        #         data_lbl = np.array(data_lbl / 255, dtype=np.float32)
        #     images.append(data_img)
        #     labels.append(data_lbl)

    while True:
        X = []
        Y = []
        b = 0
        while b < batch_size:
            index = np.random.randint(len(images))
            image = images[index]
            img_shp = image.shape
            label = labels[index]
            patch_img = np.copy(image)
            patch_lbl = np.copy(label)
            img_and_mask = np.concatenate((patch_img, patch_lbl), axis=2)
            # Horizontal Flipping
            if np.random.randint(0, 100) == 0:
                img_and_mask = img_and_mask[:, ::-1, ...]
            # Vertical Flipping
            if np.random.randint(0, 100) == 0:
                img_and_mask = img_and_mask[::-1, :, ...]
            # # Patch Stretching
            # if np.random.randint(0, 100) == 0:
            #     height_scale = 1 + np.random.rand()
            #     width_scale = 1 + np.random.rand()
            #     # Horizontal
            #     if np.random.choice([True, False]):
            #         patch_img = cv2.resize(patch_img, (int(patch_size[1] * width_scale), patch_size[0]))
            #         patch_lbl = cv2.resize(patch_lbl, (int(patch_size[1] * width_scale), patch_size[0]))
            #     # Vertical
            #     if np.random.choice([True, False]):
            #         patch_img = cv2.resize(patch_img, (patch_size[1], int(patch_size[0] * height_scale)))
            #         patch_lbl = cv2.resize(patch_lbl, (patch_size[1], int(patch_size[0] * height_scale)))
            #     if len(patch_img.shape) < 3:
            #         patch_img = np.expand_dims(patch_img, axis=-1)
            #         patch_lbl = np.expand_dims(patch_lbl, axis=-1)
            #     img_and_mask[:, :, :1] = patch_img[:patch_size[0], :patch_size[1], ...]
            #     img_and_mask[:, :, 1:] = patch_lbl[:patch_size[0], :patch_size[1], ...]

            if np.random.randint(0, 100) == 0:
                img_and_mask = random_zoom(
                    img_and_mask, zoom_range=(0.7, 1.3),
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0
                )
            if np.random.randint(0, 1000) == 0:
                img_and_mask = random_shift(        # very low prob
                    img_and_mask, wrg=1.5, hrg=1.5,
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0,
                )
            if np.random.randint(0, 100) == 0:
                img_and_mask = random_rotation(
                    img_and_mask, rg=45,
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0
                )
            if np.random.randint(0, 100) == 0:
                img_and_mask[:, :, :1] = random_brightness(
                    patch_img,
                    brightness_range=(0.8, 1.1)
                )/255
            if np.random.randint(0, 100) == 0:
                img_and_mask = random_shear(
                    img_and_mask, intensity = 45,
                    row_axis=0, col_axis=1, channel_axis=2,
                    fill_mode='constant', cval=0.0,
                )

            patch_img = img_and_mask[:, :, :1]
            patch_lbl = img_and_mask[:, :, 1:]
            if np.random.randint(0, 100) == 0:
                patch_img = add_sp_noise(patch_img)
            # Can be ignored
            if np.sum(patch_lbl) == 0 and np.random.randint(0, 100) > 0:    # 1% chance of selecting all negative sample
                continue
            # X.append(np.expand_dims(patch_img, axis=-1))
            # Y.append(np.expand_dims(patch_lbl, axis=-1))
            X.append(patch_img)
            Y.append(patch_lbl)
            # k += 1
            b += 1
        if caps:
            x, y = np.array(X), np.array(Y)
            yield ([x, y], [y, y*x])
        else:

            yield np.array(X), np.array(Y)


def main():
    current_batch = 0
    total_batches = 10
    batch_size = 32
    patch_size = (256, 256)
    pbar = tqdm(total=total_batches * batch_size, desc='Progress')
    i = 0
    for data in data_generator('testing_dataset', 'pre-processed', 'label-1', 'png', batch_size, patch_size):
        i = current_batch * batch_size
        for X, Y in zip(data[0], data[1]):
            Image.fromarray(np.array(X[:, :, 0] * 255, dtype=np.uint8)).save(RESULT_DIR + '/{:08d}_1.png'.format(i))
            Image.fromarray(np.array(Y[:, :, 0] * 255, dtype=np.uint8)).save(RESULT_DIR + '/{:08d}_2.png'.format(i))
            i += 1
            pbar.update()
        current_batch += 1
        if current_batch >= total_batches:
            break
    pbar.close()


if __name__ == '__main__':
    main()