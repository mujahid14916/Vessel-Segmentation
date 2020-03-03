from glob import glob
from pprint import pprint as pp
from PIL import Image
import numpy as np
from gen_preprocess_data import pre_process_image
from gen_preprocess_data import extract_ordered_overlap
from gen_preprocess_data import paint_border_overlap
from gen_preprocess_data import recompone_overlap
from matplotlib import pyplot as plt
import os
import cv2
from SegCaps.capsnet import CapsNetR3
from SegCaps.capsule_layers import ConvCapsuleLayer, Length, Mask, DeconvCapsuleLayer
import tensorflow as tf
from tqdm import tqdm


IMAGE_RESIZE_PER = 1         # Resize Percentage
PATCH_SIZE = (256, 256)           # (height, width)
STRIDE_SIZE = (128, 128)          # (height, width)
IMG_SIZE = None

DIR_NAME = '../retcam'
RESULT_DIR = DIR_NAME + '_caps_results_1'
MODEL_PATH = 'models/segcaps-rop-model-10-0.093461-0.893399.hdf5'


input_shape=(256, 256, 1)
train_model, test_model, manip_model = CapsNetR3(input_shape)
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

# input_shape=(256, 256, 1)
# train_model, test_model, manip_model = CapsNetR3(input_shape)
# # test_model.summary()

# pp(file_names)

# models_path = glob('models/segcaps-model-120-0.099468-0.946507.hdf5')
# for k, model_path in enumerate(models_path, 1):
#     model = tf.keras.models.load_model(model_path, 
#                                        custom_objects={
#                                            'ConvCapsuleLayer': ConvCapsuleLayer,
#                                            'Mask': Mask,
#                                            'Length': Length,
#                                            'DeconvCapsuleLayer': DeconvCapsuleLayer
#                                        }, compile=False)

#     for test_layer in test_model.layers:
#         for train_layer in model.layers:
#             if train_layer.name == test_layer.name:
#                 test_layer.set_weights(train_layer.get_weights())
#                 found = True
#         if not found:
#             print(test_layer.name)
    
#     for i, file_name in enumerate(file_names[:200], 1):
#         print('-'*80)
#         print("Progress: {}/{}".format(i, len(file_names)))
#         image = Image.open(file_name)
#         if not IMG_SIZE:
#             IMG_SIZE = (int(image.size[0] * IMAGE_RESIZE_PER), 
#                         int(image.size[1] * IMAGE_RESIZE_PER))
#         image = np.asarray(image.resize(IMG_SIZE))
#         if len(image.shape) == 3:
#             image = image[:, :, 0:3]
#         else:
#             image = np.expand_dims(image, axis=-1)

#         print(image.shape)
#         image = pre_process_image(image, gamma=0.9)

#         #extend both images and masks so they can be divided exactly by the patches dimensions
#         image = paint_border_overlap(image, *PATCH_SIZE, *STRIDE_SIZE)
#         new_size = (image.shape[2], image.shape[3])

#         print ("\ntest images/masks shape:")
#         print (image.shape)
#         print ("test images range (min-max): " +str(np.min(image)) +' - '+str(np.max(image)))
#         print ("test masks are within 0-1\n")

#         image_patches = extract_ordered_overlap(image, *PATCH_SIZE, *STRIDE_SIZE)

#         print ("\ntest PATCHES images/masks shape:")
#         print (image_patches.shape)
#         print ("test PATCHES images range (min-max): " +str(np.min(image_patches)) +' - '+str(np.max(image_patches)))
        
#         # Prediction
#         image_patches = np.einsum('klij->kijl', image_patches)
#         predictions = test_model.predict(image_patches, batch_size=1, verbose=1)[0]
#         predictions = np.einsum('kijl->klij', predictions)

#         original_image = recompone_overlap(predictions, *new_size, *STRIDE_SIZE)
#         print(original_image.shape)
#         original_image = np.einsum('klij->kijl', original_image)
#         original_image = original_image[:, 0:IMG_SIZE[1], 0:IMG_SIZE[0], :]
#         original_image = np.repeat(original_image[0], 3, axis=-1)
        
#         threshold = cv2.threshold(original_image, 150/255, 255/255, cv2.THRESH_BINARY)[1]
#         image_name = ''.join(file_name.replace('\\', '/').split('/')[-1].split('.')[:-1])
#         save_image_path = RESULT_DIR + '/' + image_name + '_' + str(PATCH_SIZE) + '_' + str(STRIDE_SIZE) + \
#                           '_{}capsa.jpg'.format('_'.join(model_path.replace('\\', '/').split('/')[-1].split('-')[:2]))
#         # plt.imsave(save_image_path, np.repeat(original_image[0], 3, axis=-1))
#         # plt.imsave(save_image_path, original_image)
#         cv2.imwrite(save_image_path, np.array(original_image*255, dtype=np.uint8))
#         cv2.imwrite('.'.join(save_image_path.split('.')[:-1]) + '_thresh.jpg', np.array(threshold*255, dtype=np.uint8))
#         print("Saving Image as", save_image_path)
#         print()

def main():
    files = glob(DIR_NAME + '/*.png')
    if not os.path.isdir(RESULT_DIR):
        os.mkdir(RESULT_DIR)
    for file in tqdm(files):
        image = tf.keras.preprocessing.image.load_img(file)
        image = np.asarray(image)
        img, th = segment_vessel_capsnet(image, 1)
        image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + '.jpg'), img)
        tf.keras.preprocessing.image.save_img(os.path.join(RESULT_DIR, image_name + '-th.jpg'), th)
        
    # print("Horizontal Flip")
    # img, th = segment_vessel_capsnet(image[:, ::-1, ...])
    # # tf.keras.preprocessing.image.save_img('../retcam_caps_results_22/1.png', image)
    # tf.keras.preprocessing.image.save_img('../2.png', th)
    
    # print("Vertical Flip")
    # img, th = segment_vessel_capsnet(image[::-1, :, ...])
    # # tf.keras.preprocessing.image.save_img('../retcam_caps_results_22/1.png', image)
    # tf.keras.preprocessing.image.save_img('../3.png', th)
    
    # print("Horizontal & Vertical Flip")
    # img, th = segment_vessel_capsnet(image[::-1, ::-1, ...])
    # # tf.keras.preprocessing.image.save_img('../retcam_caps_results_22/1.png', image)
    # tf.keras.preprocessing.image.save_img('../4.png', th)


if __name__ == '__main__':
    main()
