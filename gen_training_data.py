import cv2
import numpy as np
from glob import glob
from PIL import Image
import os


def rgb2gray(rgb):
    assert (len(rgb.shape)==4)  #4D arrays
    assert (rgb.shape[3]==3)
    bn_imgs = rgb[:,:,:,0]*0.299 + rgb[:,:,:,1]*0.587 + rgb[:,:,:,2]*0.114
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],rgb.shape[1],rgb.shape[2],1))
    return bn_imgs


def dataset_normalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


def clahe_equalized(imgs):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    #create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i,0] = clahe.apply(np.array(imgs[i,0], dtype = np.uint8))
    return imgs_equalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape)==4)  #4D arrays
    assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i,0] = cv2.LUT(np.array(imgs[i,0], dtype = np.uint8), table)
    return new_imgs


def pre_process_image(data):
    assert(len(data.shape)==4)
    assert (data.shape[1]==3)  #Use the original images
    #black-white conversion
    train_imgs = rgb2gray(data)
    #my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs/255.  #reduce to 0-1 range
    return train_imgs


def read_training_images(files):
    images = []
    for i, file in enumerate(files, 1):
        print("{}/{}: {}".format(i, len(files), file))
        image = Image.open(file)
        images.append(np.asarray(image)[:, :, :3])
    return images


def main():
    files = glob('../neo/*png')
    result_dir = '../neo/pre-processed/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    images = read_training_images(files)
    for file, image in zip(files, images):
        image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        print(image.shape)
        gray_scale = rgb2gray(np.expand_dims(image, axis=0))[0]
        print(gray_scale.shape)
        out = np.repeat(gray_scale, repeats=[3], axis=-1)
        print(out.shape)
        cv2.imwrite(result_dir + image_name + '_1_gray.jpg', out)

        gray_scale = np.einsum('kijl->klij', np.expand_dims(gray_scale, axis=0))
        normalized = dataset_normalized(gray_scale)
        normalized = np.einsum('klij->kijl', normalized)[0]
        print(out.shape)
        cv2.imwrite(result_dir + image_name + '_2_norm.jpg', normalized)

        normalized = np.einsum('kijl->klij', np.expand_dims(normalized, axis=0))
        clache = clahe_equalized(normalized)
        clache = np.einsum('klij->kijl', clache)[0]
        print(clache.shape)
        out = np.repeat(clache, repeats=[3], axis=-1)
        print(out.shape)
        cv2.imwrite(result_dir + image_name + '_3_clache.jpg', out)

        clache = np.einsum('kijl->klij', np.expand_dims(clache, axis=0))
        gamma = adjust_gamma(clache, 1.2)
        gamma = np.einsum('klij->kijl', gamma)[0]
        print(gamma.shape)
        out = np.repeat(gamma, repeats=[3], axis=-1)
        print(out.shape)
        cv2.imwrite(result_dir + image_name + '_4_gamma.jpg', out)
        # break


if __name__ == '__main__':
    main()
