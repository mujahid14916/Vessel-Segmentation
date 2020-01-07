from glob import glob
from PIL import Image
# from SegCaps.model_helper import create_model
import os
from pre_process import pre_process_image
import numpy as np

PATCH_SIZE = (64, 64)
BATCH_SIZE = 64
WEIGHT_FILE = 'models/segcaps_01.hdf5'

NET_INPUT_SHAPE = (64, 64, 1)
images_to_test = glob('training_dataset/input/*.jpg')


# train_model, eval_model, manup_model = create_model(net='segcapsr3', input_shape=NET_INPUT_SHAPE)
# if os.path.isfile(WEIGHT_FILE):
#     eval_model.load_weights(WEIGHT_FILE)
# else:
#     print("Model not found, Cannot Infer")
#     exit()


for image_path in images_to_test:
    print('-'*80)
    original_img = np.asarray(Image.open(image_path))
    print(original_img.shape)
    input_img = pre_process_image(original_img)
    orig_img_shape = input_img.shape
    output_img_shape = (int(np.ceil(input_img.shape[0]/PATCH_SIZE[0]) * PATCH_SIZE[0]), 
                        int(np.ceil(input_img.shape[1]/PATCH_SIZE[1]) * PATCH_SIZE[1]),
                        input_img.shape[2])
    output_img = np.zeros(output_img_shape)
    patch_inputs = []
    for i in range(0, input_img.shape[0], PATCH_SIZE[0]):
        for j in range(0, input_img.shape[1], PATCH_SIZE[1]):
            net_input = input_img[i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1], :]
            patch_shape = net_input.shape
            if patch_shape[0] != PATCH_SIZE[0]:
                net_input = np.append(net_input, 
                                      np.zeros((PATCH_SIZE[0] - patch_shape[0], patch_shape[1], patch_shape[2])), 
                                      axis=0)
            if patch_shape[1] != PATCH_SIZE[1]:
                net_input = np.append(net_input,
                                      np.zeros((PATCH_SIZE[0], PATCH_SIZE[1] - patch_shape[1], patch_shape[2])),
                                      axis=1)
            patch_inputs.append(net_input)
    batch_input = np.array(patch_inputs)
    print(batch_input.shape)
    # predicted_mask = eval_model.predict(batch_input)
    k = 0
    for i in range(0, input_img.shape[0], PATCH_SIZE[0]):
        for j in range(0, input_img.shape[1], PATCH_SIZE[1]):
            # output_img[i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1], :] = predicted_mask[k]
            output_img[i:i+PATCH_SIZE[0], j:j+PATCH_SIZE[1], :] = batch_input[k]
            k += 1
    print(output_img.shape)
    output_img = output_img[:orig_img_shape[0], :orig_img_shape[1], :]
    print(output_img.shape)
    # break
    print()


