import cv2
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


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


# def pre_process_image(data):
#     assert(len(data.shape)==4)
#     assert (data.shape[1]==3)  #Use the original images
#     #black-white conversion
#     train_imgs = rgb2gray(data)
#     #my preprocessing:
#     train_imgs = dataset_normalized(train_imgs)
#     train_imgs = clahe_equalized(train_imgs)
#     train_imgs = adjust_gamma(train_imgs, 1.2)
#     train_imgs = train_imgs/255.  #reduce to 0-1 range
#     return train_imgs


def read_training_images(files):
    images = []
    for i in tqdm(range(len(files)), desc="Reading Images"):
        file = files[i]
        image = Image.open(file)
        images.append(np.asarray(image))
    return images


def pre_process_image(image):
    gray_scale = rgb2gray(np.expand_dims(image, axis=0))[0]
    normalized = cv2.normalize(gray_scale, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) * 255
    normalized = np.einsum('kijl->klij', np.reshape(normalized, (1, *normalized.shape, 1)))
    clache = clahe_equalized(normalized)
    gamma = adjust_gamma(clache, 1.2)
    out = np.einsum('klij->kijl', gamma)[0]
    out = np.repeat(out, repeats=[3], axis=-1)
    return out


def main():
    input_files = glob('training_dataset/input/*jpg')
    result_dir = 'training_dataset/pre-processed/'
    images = read_training_images(input_files)
    processed = []
    for i in tqdm(range(len(input_files)), desc="Processing Images"):
        file, image = input_files[i], images[i]
        image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        out = np.array(pre_process_image(image), dtype=np.uint8)
        # cv2.imwrite(result_dir + image_name + '.png', out)
        Image.fromarray(out).convert('L').save(result_dir + image_name + '.png')
        processed.append(out)

if __name__ == '__main__':
    main()
