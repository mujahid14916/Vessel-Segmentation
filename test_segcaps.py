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
from SegCaps.capsnet import CapsNetR3
from SegCaps.capsule_layers import ConvCapsuleLayer, Length, Mask, DeconvCapsuleLayer
import tensorflow as tf


IMAGE_RESIZE_PER = 0.25         # Resize Percentage
PATCH_SIZE = (256, 256)           # (height, width)
STRIDE_SIZE = (64, 64)          # (height, width)
IMG_SIZE = None

DIR_NAME = '../neo'
RESULT_DIR = DIR_NAME + '_caps_result'


input_shape=(256, 256, 1)
train_model, test_model, manip_model = CapsNetR3(input_shape)
# test_model.summary()

file_names = glob(DIR_NAME + '/*.png')
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
pp(file_names)

models_path = glob('models/segcaps-model-15*.hdf5')
for k, model_path in enumerate(models_path, 1):
    model = tf.keras.models.load_model(model_path, 
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
                found = True
        if not found:
            print(test_layer.name)
    
    for i, file_name in enumerate(file_names, 1):
        print('-'*80)
        print("Progress: {}/{}".format(i, len(file_names)))
        image = Image.open(file_name)
        if not IMG_SIZE:
            IMG_SIZE = (int(image.size[0] * IMAGE_RESIZE_PER), 
                        int(image.size[1] * IMAGE_RESIZE_PER))
        image = np.asarray(image.resize(IMG_SIZE))
        image = image[:, :, 0:3]

        print(image.shape)
        image = pre_process_image(image, gamma=0.9)

        #extend both images and masks so they can be divided exactly by the patches dimensions
        image = paint_border_overlap(image, *PATCH_SIZE, *STRIDE_SIZE)
        new_size = (image.shape[2], image.shape[3])

        print ("\ntest images/masks shape:")
        print (image.shape)
        print ("test images range (min-max): " +str(np.min(image)) +' - '+str(np.max(image)))
        print ("test masks are within 0-1\n")

        image_patches = extract_ordered_overlap(image, *PATCH_SIZE, *STRIDE_SIZE)

        print ("\ntest PATCHES images/masks shape:")
        print (image_patches.shape)
        print ("test PATCHES images range (min-max): " +str(np.min(image_patches)) +' - '+str(np.max(image_patches)))
        
        # Prediction
        image_patches = np.einsum('klij->kijl', image_patches)
        predictions = test_model.predict(image_patches, batch_size=1, verbose=1)[0]
        predictions = np.einsum('kijl->klij', predictions)

        orinal_image = recompone_overlap(predictions, *new_size, *STRIDE_SIZE)
        print(orinal_image.shape)
        orinal_image = np.einsum('klij->kijl', orinal_image)
        orinal_image = orinal_image[:, 0:IMG_SIZE[1], 0:IMG_SIZE[0], :]
        image_name = ''.join(file_name.replace('\\', '/').split('/')[-1].split('.')[:-1])
        save_image_path = RESULT_DIR + '/' + image_name + '_' + str(PATCH_SIZE) + '_' + str(STRIDE_SIZE) + \
                          '_{}g.jpg'.format('_'.join(model_path.replace('\\', '/').split('/')[-1].split('-')[:2]))
        plt.imsave(save_image_path, np.repeat(orinal_image[0], 3, axis=-1))
        print("Saving Image as", save_image_path)
        print()