from glob import glob
import tensorflow as tf
import numpy as np
from EyeClassification import classifier as eye_classifier, helper_utils
from EyeFilter import classifier as eye_filter


files = glob('../retcam/*.png')

files_dict = {}

for file in files:
    file_name = '.'.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    file_prefix = file_name.split('_')[0]
    if file_prefix in files_dict:
        files_dict[file_prefix].append({'path': file, 'name': file_name})
    else:
        files_dict[file_prefix] = [{'path': file, 'name': file_name}]
    
for key in files_dict:
    images = []
    invalid_index = []
    for i, fundus in enumerate(files_dict[key]):
        image = tf.keras.preprocessing.image.load_img(fundus['path'])
        image = np.asarray(image)
        if helper_utils.is_fundus(image):
            images.append(image)
        else:
            invalid_index.append(i)

    valid = []
    for i, d in enumerate(files_dict[key]):
        if i not in invalid_index:
            valid.append(d)
    files_dict[key] = valid

    images = np.array(images)
    print(images.shape)
    # Eye Position
    classes = eye_classifier.classify(images)
    assert len(files_dict[key]) == len(classes)
    # TODO: Add Seperator based on Intuition

    left = []
    right = []
    for i, pos in enumerate(classes):
        if pos == 'OD':
            right.append(images[i])
        elif pos == 'OS':
            left.append(images[i])
    left = np.array(left)
    right = np.array(right)
    print(left.shape)
    print(right.shape)


    # Filter Bad Eye
    valid_right = eye_filter.classify(right)
    valid_left = eye_filter.classify(left)
    print(valid_left.shape)
    print(valid_right.shape)

    # Get Vessel Segment

    exit()