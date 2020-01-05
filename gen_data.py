from matplotlib import pyplot as plt
from glob import glob
import numpy as np
import os
import cv2
from PIL import Image
from tqdm import tqdm


PATCH_SIZE = (256, 256)       # (height, width)
TOTAL_PATCHES = 500
IMG_MAX_HEIGHT = 600
IMG_MIN_HEIGHT = 300
RESULT_DIR = 'training_dataset/patches'
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)


def data_generator(dataset_root_dir, image_dir, label_dir, image_ext, batch_size, patch_size=(64, 64), image_min_max_hgt=(300, 600)):
    k = 0
    files = glob('{}/{}/*.{}'.format(dataset_root_dir, image_dir, image_ext))
    images = []
    labels = []
    for file in files:
        img = np.asarray(Image.open(file))
        lbl = np.asarray(Image.open(file.replace(image_dir, label_dir)))
        if np.max(img) > 1:
            data_img = img / 255
        if np.max(lbl) > 1:
            data_lbl = lbl / 255
        images.append(data_img)
        labels.append(data_lbl)

        # Append Scaled Images
        for _ in range(5):
            img_height = np.random.randint(low=image_min_max_hgt[0], high=image_min_max_hgt[1]+1)
            img_width = int(img.shape[1] / img.shape[0] * img_height)   # original_width / original_height * height
            data_img = cv2.resize(img, (img_width, img_height))     # OpenCV (width, height) format
            data_lbl = cv2.resize(lbl, (img_width, img_height))
            if np.max(data_img) > 1:
                data_img = data_img / 255
            if np.max(data_lbl) > 1:
                data_lbl = data_lbl / 255
            images.append(data_img)
            labels.append(data_lbl)

    # pbar = tqdm(total=TOTAL_PATCHES, desc='Patch Progress')
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
            patch_img = image[rnd_height:rnd_height+patch_size[0], rnd_width:rnd_width+patch_size[1]]
            patch_lbl = label[rnd_height:rnd_height+patch_size[0], rnd_width:rnd_width+patch_size[1]]
            # Horizontal Flipping
            if np.random.randint(0, 1000) == 0:
                patch_img = patch_img[:, ::-1]
                patch_lbl = patch_lbl[:, ::-1]
            # Vertical Flipping
            if np.random.randint(0, 1000) == 0:
                patch_img = patch_img[::-1, :]
                patch_lbl = patch_lbl[::-1, :]
            # Can be ignored
            if np.sum(patch_lbl) == 0:
                # print("Skipped, k =", k)
                # Image.fromarray(patch_img).save(RESULT_DIR + '/SKIPPED_{}_1.png'.format(k))
                # Image.fromarray(patch_lbl).save(RESULT_DIR + '/SKIPPED_{}_2.png'.format(k))
                continue
            X.append(np.expand_dims(patch_img, axis=-1))
            Y.append(np.expand_dims(patch_lbl, axis=-1))
            # Image.fromarray(patch_img).save(RESULT_DIR + '/{:08d}_1.png'.format(k))
            # Image.fromarray(patch_lbl).save(RESULT_DIR + '/{:08d}_2.png'.format(k))
            b += 1
            # k += 1
        yield np.array(X), np.array(Y)
    #     pbar.update()
    #     if k >= TOTAL_PATCHES:
    #         break
    # pbar.close()


def main():
    x = 0
    total = 1000000
    pbar = tqdm(total=total, desc='Progress')
    for data in data_generator('training_dataset', 'pre-processed', 'label-1', 'png', 32):
        x += 1
        pbar.update()
        if x > total:
            break

if __name__ == '__main__':
    main()