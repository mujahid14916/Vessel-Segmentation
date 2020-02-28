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
    file_idx = file_name.split('_')[1]
    if len(file_idx) < 2:
        file_idx = '{}{}'.format('0'*(2 - len(file_idx)), file_idx)
    if file_prefix in files_dict:
        files_dict[file_prefix].append({'path': file, 'name': file_prefix + '_' + file_idx})
    else:
        files_dict[file_prefix] = [{'path': file, 'name': file_prefix + '_' + file_idx}]
    files_dict[file_prefix] = sorted(files_dict[file_prefix], key=lambda x: x['name'])


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
    print(classes)
    assert len(files_dict[key]) == len(classes)
    # TODO: Add Seperator based on Intuition

    left = []
    right = []
    right_mapping = []
    left_mapping = []
    for i, pos in enumerate(classes):
        if pos == 'OD':
            right.append(images[i])
            right_mapping.append(i)
        elif pos == 'OS':
            left.append(images[i])
            left_mapping.append(i)
    left = np.array(left)
    right = np.array(right)
    print(left.shape)
    print(right.shape)


    # Filter Bad Eye
    valid_right = eye_filter.classify(right)
    valid_left = eye_filter.classify(left)

    for v in valid_right:
        if v == 0 and v in right_mapping:
            right_mapping.remove(v)

    for v in valid_left:
        if v == 0 and v in left_mapping:
            left_mapping.remove(v)

    # Get Vessel Segment
