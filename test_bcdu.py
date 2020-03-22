from glob import glob
from pprint import pprint as pp
from PIL import Image
import numpy as np
from pre_process import pre_process_image
from pre_process import extract_ordered_overlap
from pre_process import paint_border_overlap
from pre_process import recompone_overlap
import BCDU.models as M
import os
import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage
import cv2


IMAGE_RESIZE_PER = 0.25         # Resize Percentage
PATCH_SIZE = (128, 128)           # (height, width)
STRIDE_SIZE = (64, 64)          # (height, width)
IMG_SIZE = None

DIR_NAME = '../neo'
RESULT_DIR = DIR_NAME + '_40_bcdu_rotation'
MODEL_PATH = 'models/bcdu_weight_dice-45-0.920427.hdf5'


model = M.BCDU_net_D3(input_size = (*PATCH_SIZE, 1))
model.load_weights(MODEL_PATH)

def rotate_image(image, deg=45):
    return ndimage.rotate(np.asarray(image), deg, reshape=True)


def undo_rotate_image(image, deg=45, shape=None):
    # TODO: Fix for degree greater than 80
    if shape is None:
        pi_factor = np.pi/180
        try:
            a = np.array([[np.sin((90 - deg)*pi_factor), np.sin(deg*pi_factor)], 
                        [np.cos((90 - deg)*pi_factor), np.cos(deg*pi_factor)]])
            a = np.abs(a)
            r = np.array([[image.shape[0]], [image.shape[1]]])
            img_height, img_width = np.array(np.squeeze(np.matmul(np.linalg.inv(a), r)), dtype=np.int)
        except:
            raise ValueError("Shape Cannot be determined, please provide shape parameter")
    else:
        img_height, img_width = shape[:2]
    rotated_image = ndimage.rotate(image, -deg, reshape=False)
    h = abs(rotated_image.shape[0] - img_height)//2
    w = abs(rotated_image.shape[1] - img_width)//2
    return rotated_image[h:h+img_height, w:w+img_width]


def segment_vessel_bcdu(image, image_scale_percentage=1, th_value=150):
    img = np.copy(image)
    if np.max(img) > 1:
        img = np.array(img/255., dtype=np.float)
    img_size  = (int(img.shape[0] * image_scale_percentage), 
                 int(img.shape[1] * image_scale_percentage))
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img = tf.image.resize(img, img_size)
    img = img[:, :, 0:3]

    img = pre_process_image(img, gamma=0.9)

    #extend both images and masks so they can be divided exactly by the patches dimensions
    img = paint_border_overlap(img, *PATCH_SIZE, *STRIDE_SIZE, verbose=False)
    new_size = (img.shape[2], img.shape[3])

    image_patches = extract_ordered_overlap(img, *PATCH_SIZE, *STRIDE_SIZE, verbose=False)
    # Prediction
    image_patches = np.einsum('klij->kijl', image_patches)
    predictions = model.predict(image_patches, batch_size=16)
    predictions = np.einsum('kijl->klij', predictions)

    original_image = recompone_overlap(predictions, *new_size, *STRIDE_SIZE, verbose=False)
    original_image = np.einsum('klij->kijl', original_image)
    original_image = original_image[0, 0:img_size[0], 0:img_size[1], :]
    rgb_image = np.repeat(original_image, 3, axis=-1)
    threshold = cv2.threshold(rgb_image, th_value/255, 255/255, cv2.THRESH_BINARY)[1]
    
    return original_image, threshold


def main():
    files = glob(DIR_NAME + '/*.png')
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    for file in tqdm(files[775:]):
        image = tf.keras.preprocessing.image.load_img(file)
        image = np.asarray(image)
        img, th = segment_vessel_bcdu(image, IMAGE_RESIZE_PER, 150)
        (h, w) = img.shape[:2]
        res = np.array(img, dtype=np.float64)
        res_th = np.array(th, dtype=np.float64)
        image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx.jpg'), img)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-th.jpg'), th)
        
    # print("Horizontal Flip")
        img, th = segment_vessel_bcdu(image[:, ::-1, ...], IMAGE_RESIZE_PER, 150)
        res += img[:, ::-1, ...]
        res_th += th[:, ::-1, ...]
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-h.jpg'), img[:, ::-1, ...])
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-h-th.jpg'), th[:, ::-1, ...])
    
    # print("Vertical Flip")
        img, th = segment_vessel_bcdu(image[::-1, :, ...], IMAGE_RESIZE_PER, 150)
        res += img[::-1, :, ...]
        res_th += th[::-1, :, ...]
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-v.jpg'), img[::-1, :, ...])
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-v-th.jpg'), th[::-1, :, ...])
    
    # print("Horizontal & Vertical Flip")
        img, th = segment_vessel_bcdu(image[::-1, ::-1, ...], IMAGE_RESIZE_PER, 150)
        res += img[::-1, ::-1, ...]
        res_th += th[::-1, ::-1, ...]
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-hv.jpg'), img[::-1, ::-1, ...])
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-hv-th.jpg'), th[::-1, ::-1, ...])
        
    # print("Rotate by 30 deg")
        deg = 30
        img, th = segment_vessel_bcdu(rotate_image(image, deg=deg), IMAGE_RESIZE_PER, 150)
        r = undo_rotate_image(img, deg=deg, shape=(h, w))
        r[r > 1] = 1
        r[r < 0] = 0
        t = undo_rotate_image(th, deg=deg, shape=(h, w))
        t[t > 1] = 1
        t[t < 0] = 0
        res += r
        res_th += t
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-30.jpg'), r)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-30-th.jpg'), t)
        
    # print("Rotate by 45 deg")
        deg = 45
        img, th = segment_vessel_bcdu(rotate_image(image, deg=deg), IMAGE_RESIZE_PER, 150)
        r = undo_rotate_image(img, deg=deg, shape=(h, w))
        r[r > 1] = 1
        r[r < 0] = 0
        t = undo_rotate_image(th, deg=deg, shape=(h, w))
        t[t > 1] = 1
        t[t < 0] = 0
        res += r
        res_th += t
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-45.jpg'), r)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-45-th.jpg'), t)
        
    # print("Rotate by 60 deg")
        deg = 60
        img, th = segment_vessel_bcdu(rotate_image(image, deg=deg), IMAGE_RESIZE_PER, 150)
        r = undo_rotate_image(img, deg=deg, shape=(h, w))
        r[r > 1] = 1
        r[r < 0] = 0
        t = undo_rotate_image(th, deg=deg, shape=(h, w))
        t[t > 1] = 1
        t[t < 0] = 0
        res += r
        res_th += t
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-60.jpg'), r)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-60-th.jpg'), t)
        
        res = np.array(np.clip(res*255, 0, 255), dtype=np.uint8)
        res_th = np.array(np.clip(res_th*255, 0, 255), dtype=np.uint8)
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-combined.jpg'), res)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-combined-th.jpg'), res_th)

if __name__ == '__main__':
    main()
