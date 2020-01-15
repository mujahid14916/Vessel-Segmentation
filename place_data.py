import os
from glob import glob
import shutil
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2

training_dir = 'training_dataset'

input_dir = training_dir + '/input'
processed_dir = training_dir + '/pre-processed'
label_1_dir = training_dir + '/label-1'
label_2_dir = training_dir + '/label-2'


if not os.path.isdir(training_dir):
    os.mkdir(training_dir)

if not os.path.isdir(input_dir):
    os.mkdir(input_dir)

if not os.path.isdir(processed_dir):
    os.mkdir(processed_dir)

if not os.path.isdir(label_1_dir):
    os.mkdir(label_1_dir)

if not os.path.isdir(label_2_dir):
    os.mkdir(label_2_dir)


# CHASE DB
# ------------------------------------------------------------------------------------------------
ChaseDB_dir_path = 'Dataset/CHASEDB1/'
chase_files_labels = glob(ChaseDB_dir_path + '*.png')
chase_files_input = glob(ChaseDB_dir_path + '*.jpg')

for i, file in enumerate(chase_files_input, 1):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    shutil.copy(file, input_dir + '/CHASE_{:02d}.jpg'.format(i))
    # shutil.copy(ChaseDB_dir_path + image_name + '_1stHO.png', label_1_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(ChaseDB_dir_path + image_name + '_2ndHO.png', label_2_dir + '/CHASE_{:02d}.png'.format(i))
    label_1 = Image.open(ChaseDB_dir_path + image_name + '_1stHO.png')
    label_2 = Image.open(ChaseDB_dir_path + image_name + '_2ndHO.png')
    label_1.convert("L").save(label_1_dir + '/CHASE_{:02d}.png'.format(i))
    label_2.convert("L").save(label_2_dir + '/CHASE_{:02d}.png'.format(i))


# STARE
# ------------------------------------------------------------------------------------------------
Stare_dir_path = 'Dataset/STARE/training-set/'
Stare_label_1_dir = 'Dataset/STARE/labels-ah-jpg/'
Stare_label_2_dir = 'Dataset/STARE/labels-vk-jpg/'
stare_files = glob(Stare_dir_path + '*.jpg')

for i, file in enumerate(stare_files, 1):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    shutil.copy(file, input_dir + '/STARE_{:02d}.jpg'.format(i))
    # shutil.copy(Stare_label_1_dir + image_name + 'ah.jpg', label_1_dir + '/STARE_{:02d}.jpg'.format(i))
    # shutil.copy(Stare_label_2_dir + image_name + 'vk.jpg', label_2_dir + '/STARE_{:02d}.jpg'.format(i))
    label_1 = Image.open(Stare_label_2_dir + image_name + 'vk.jpg')
    label_2 = Image.open(Stare_label_1_dir + image_name + 'ah.jpg')
    label_1.convert('L').save(label_1_dir + '/STARE_{:02d}.png'.format(i))
    label_2.convert('L').save(label_2_dir + '/STARE_{:02d}.png'.format(i))


# DRIVE
# ------------------------------------------------------------------------------------------------
drive_training_dir = 'Dataset/DRIVE/training/images/'
drive_training_label_dir = 'Dataset/DRIVE/training/1st_manual/'
drive_training_files = glob(drive_training_dir + '*.tif')

k = 1
for i, file in enumerate(drive_training_files, k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    shutil.copy(file, input_dir + '/DRIVE_{:02d}.jpg'.format(i))
    # shutil.copy(drive_training_label_dir + image_name.replace('training', 'manual1') + '.gif', label_1_dir + '/DRIVE_{:02d}.gif'.format(i))
    label_1 = Image.open(drive_training_label_dir + image_name.replace('training', 'manual1') + '.gif')
    label_1.convert('L').save(label_1_dir + '/DRIVE_{:02d}.png'.format(i))
    k += 1
# ------------------------------------------------------------------------------------------------
drive_test_dir = 'Dataset/DRIVE/test/images/'
drive_test_label_1_dir = 'Dataset/DRIVE/test/1st_manual/'
drive_test_label_2_dir = 'Dataset/DRIVE/test/2nd_manual/'
drive_test_files = glob(drive_test_dir + '*.tif')

for i, file in enumerate(drive_test_files, k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    shutil.copy(file, input_dir + '/DRIVE_{:02d}.jpg'.format(i))
    label_1 = Image.open(drive_test_label_2_dir + image_name.replace('test', 'manual2') + '.gif')
    label_2 = Image.open(drive_test_label_1_dir + image_name.replace('test', 'manual1') + '.gif')
    # shutil.copy(drive_test_label_1_dir + image_name.replace('test', 'manual1') + '.gif', label_1_dir + '/DRIVE_{:02d}.gif'.format(i))
    # shutil.copy(drive_test_label_2_dir + image_name.replace('test', 'manual2') + '.gif', label_2_dir + '/DRIVE_{:02d}.gif'.format(i))
    label_1.convert('L').save(label_1_dir + '/DRIVE_{:02d}.png'.format(i))
    label_2.convert('L').save(label_2_dir + '/DRIVE_{:02d}.png'.format(i))
