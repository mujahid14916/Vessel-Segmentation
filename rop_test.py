from glob import glob
from pprint import pprint as pp
from PIL import Image
import numpy as np
from pre_process import pre_process_image
from pre_process import extract_ordered_overlap
from pre_process import paint_border_overlap
from pre_process import recompone_overlap
import BCDU.models as M
from matplotlib import pyplot as plt
import os


IMAGE_RESIZE_PER = 1         # Resize Percentage
PATCH_SIZE = (64, 64)           # (height, width)
STRIDE_SIZE = (50, 50)          # (height, width)
IMG_SIZE = None

DIR_NAME = '../retcam'
RESULT_DIR = DIR_NAME + '_result'

file_names = glob(DIR_NAME + '/*.png')
if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)
pp(file_names)

model = M.BCDU_net_D3(input_size = (*PATCH_SIZE, 1))
model.load_weights('models/bcdu_weight-05-0.867993.hdf5')

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
    image = pre_process_image(image)

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
    predictions = model.predict(image_patches, batch_size=16, verbose=1)
    predictions = np.einsum('kijl->klij', predictions)

    orinal_image = recompone_overlap(predictions, *new_size, *STRIDE_SIZE)
    print(orinal_image.shape)
    orinal_image = np.einsum('klij->kijl', orinal_image)
    orinal_image = orinal_image[:, 0:IMG_SIZE[1], 0:IMG_SIZE[0], :]
    image_name = ''.join(file_name.replace('\\', '/').split('/')[-1].split('.')[:-1])
    save_image_path = RESULT_DIR + '/' + image_name + '_' + str(PATCH_SIZE) + '_' + str(STRIDE_SIZE) + '.jpg'
    plt.imsave(save_image_path, np.repeat(orinal_image[0], 3, axis=-1))
    print("Saving Image as", save_image_path)
    print()
