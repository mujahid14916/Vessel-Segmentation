from glob import glob
from pprint import pprint as pp
from PIL import Image
import numpy as np
from pre_process import pre_process_image
from pre_process import extract_ordered_overlap
from pre_process import paint_border_overlap
from pre_process import recompone_overlap
from matplotlib import pyplot as plt
import os
import cv2
from SegCaps.capsnet import CapsNetR3, CapsNetR4
from SegCaps.capsule_layers import ConvCapsuleLayer, Length, Mask, DeconvCapsuleLayer
import tensorflow as tf
from tqdm import tqdm
from scipy import ndimage
from patch_generator import square_frame


IMAGE_RESIZE_PER = 1         # Resize Percentage
PATCH_SIZE = (256, 256)           # (height, width)
STRIDE_SIZE = (128, 128)          # (height, width)
IMG_SIZE = None

DIR_NAME = '../neo'
RESULT_DIR = DIR_NAME + '_caps_results_full_150'
MODEL_PATH = 'models/segcaps-full-model-150-0.162590-0.890193.hdf5'


input_shape=(512, 512, 1)
train_model, test_model, manip_model = CapsNetR4(input_shape)
model = tf.keras.models.load_model(MODEL_PATH, 
                                    custom_objects={
                                        'ConvCapsuleLayer': ConvCapsuleLayer,
                                        'Mask': Mask,
                                        'Length': Length,
                                        'DeconvCapsuleLayer': DeconvCapsuleLayer
                                    }, compile=False)

for test_layer in test_model.layers:
    for train_layer in model.layers:
        if train_layer.name == test_layer.name:
            test_layer.set_weights(train_layer.get_weights())


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


def segment_vessel_capsnet(image, image_scale_percentage=1, th_value=150):
    if np.max(image) > 1:
        image = np.array(image/255., dtype=np.float)
    img_size  = (int(image.shape[0] * image_scale_percentage), 
                 int(image.shape[1] * image_scale_percentage))
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    image = tf.image.resize(image, img_size)
    image = image[:, :, 0:3]

    image = pre_process_image(image, gamma=0.9)

    #extend both images and masks so they can be divided exactly by the patches dimensions
    image = paint_border_overlap(image, *PATCH_SIZE, *STRIDE_SIZE, verbose=False)
    new_size = (image.shape[2], image.shape[3])

    image_patches = extract_ordered_overlap(image, *PATCH_SIZE, *STRIDE_SIZE, verbose=False)
    # Prediction
    image_patches = np.einsum('klij->kijl', image_patches)
    predictions = test_model.predict(image_patches, batch_size=1)[0]
    predictions = np.einsum('kijl->klij', predictions)

    original_image = recompone_overlap(predictions, *new_size, *STRIDE_SIZE, verbose=False)
    original_image = np.einsum('klij->kijl', original_image)
    original_image = original_image[0, 0:img_size[0], 0:img_size[1], :]
    rgb_image = np.repeat(original_image, 3, axis=-1)
    threshold = cv2.threshold(rgb_image, th_value/255, 255/255, cv2.THRESH_BINARY)[1]
    
    return original_image, threshold


def segment_vessel_full_capsnet(image, image_scale_percentage=1, th_value=150):
    h, w = image.shape[:2]
    image = square_frame(image)
    if np.max(image) > 1:
        image = np.array(image/255., dtype=np.float)
    img_size  = (512, 512)
    hr, wr = img_size[0]/h, img_size[1]/w
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    image = tf.image.resize(image, img_size)
    image = image[:, :, 0:3]

    image = pre_process_image(image, gamma=0.9)

    image_patches = np.einsum('klij->kijl', image)
    pred = test_model.predict(image_patches, batch_size=1)[0][0]
    if h != w:
        if h > w:
            diff = int((img_size[1] - img_size[0]*w/h)/2)
            pred = pred[:, diff:diff+int(w*img_size[0]/h)]
        else:
            diff = int((img_size[0] - img_size[1]*h/w)/2)
            pred = pred[diff:diff+int(h*img_size[1]/w), :]
    rgb_image = np.repeat(pred, 3, axis=-1)
    threshold = cv2.threshold(rgb_image, th_value/255, 255/255, cv2.THRESH_BINARY)[1]
    
    return tf.image.resize(pred, (h, w)), tf.image.resize(threshold, (h, w))


def main():
    files = glob(DIR_NAME + '/*.png')
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    for file in tqdm(files):
        image = tf.keras.preprocessing.image.load_img(file)
        image = np.asarray(image)
        img, th = segment_vessel_full_capsnet(image, 1, 150)
        res = np.array(img, dtype=np.float64)
        res_th = np.array(th, dtype=np.float64)
        image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx.jpg'), img)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-th.jpg'), th)
        
    # print("Horizontal Flip")
        img, th = segment_vessel_full_capsnet(image[:, ::-1, ...], 1, 150)
        res += img[:, ::-1, ...]
        res_th += th[:, ::-1, ...]
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-h.jpg'), img[:, ::-1, ...])
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-h-th.jpg'), th[:, ::-1, ...])
    
    # print("Vertical Flip")
        img, th = segment_vessel_full_capsnet(image[::-1, :, ...], 1, 150)
        res += img[::-1, :, ...]
        res_th += th[::-1, :, ...]
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-v.jpg'), img[::-1, :, ...])
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-v-th.jpg'), th[::-1, :, ...])
    
    # print("Horizontal & Vertical Flip")
        img, th = segment_vessel_full_capsnet(image[::-1, ::-1, ...], 1, 150)
        res += img[::-1, ::-1, ...]
        res_th += th[::-1, ::-1, ...]
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-hv.jpg'), img[::-1, ::-1, ...])
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-hv-th.jpg'), th[::-1, ::-1, ...])
        
    # # print("Rotate by 30 deg")
        deg = 30
        img, th = segment_vessel_full_capsnet(rotate_image(image, deg=deg), 1, 150)
        r = undo_rotate_image(img, deg=deg, shape=image.shape)
        r[r > 1] = 1
        r[r < 0] = 0
        t = undo_rotate_image(th, deg=deg, shape=image.shape)
        t[t > 1] = 1
        t[t < 0] = 0
        res += r
        res_th += t
    #     tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-30.jpg'), r)
    #     tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-30-th.jpg'), t)
        
    # # print("Rotate by 45 deg")
        deg = 45
        img, th = segment_vessel_full_capsnet(rotate_image(image, deg=deg), 1, 150)
        r = undo_rotate_image(img, deg=deg, shape=image.shape)
        r[r > 1] = 1
        r[r < 0] = 0
        t = undo_rotate_image(th, deg=deg, shape=image.shape)
        t[t > 1] = 1
        t[t < 0] = 0
        res += r
        res_th += t
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-45.jpg'), r)
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-45-th.jpg'), t)
        
    # print("Rotate by 60 deg")
        deg = 60
        img, th = segment_vessel_full_capsnet(rotate_image(image, deg=deg), 1, 150)
        r = undo_rotate_image(img, deg=deg, shape=image.shape)
        r[r > 1] = 1
        r[r < 0] = 0
        t = undo_rotate_image(th, deg=deg, shape=image.shape)
        t[t > 1] = 1
        t[t < 0] = 0
        # res += r
        # res_th += t
    #     tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-60.jpg'), r)
    #     tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-deg-60-th.jpg'), t)
        
        res = np.array(np.clip(res*255, 0, 255), dtype=np.uint8)
        res_th = np.array(np.clip(res_th*255, 0, 255), dtype=np.uint8)
        # tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-combined.jpg'), res)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + 'xxx-combined-th.jpg'), res_th)

if __name__ == '__main__':
    main()
