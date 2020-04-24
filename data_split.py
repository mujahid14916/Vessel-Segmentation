import os
from glob import glob
import shutil
from PIL import Image
import numpy as np
import cv2

training_dir = 'training_dataset'
testing_dir = 'testing_dataset'

training_input_dir = training_dir + '/input'
testing_input_dir = testing_dir + '/input'
training_processed_dir = training_dir + '/pre-processed'
testing_processed_dir = testing_dir + '/pre-processed'
training_label_1_dir = training_dir + '/label-1'
training_label_2_dir = training_dir + '/label-2'
testing_label_1_dir = testing_dir + '/label-1'
testing_label_2_dir = testing_dir + '/label-2'


if os.path.isdir(training_dir):
    shutil.rmtree(training_dir)
if os.path.isdir(testing_dir):
    shutil.rmtree(testing_dir)

os.mkdir(training_dir)
os.mkdir(training_input_dir)
os.mkdir(testing_dir)
os.mkdir(testing_input_dir)
os.mkdir(training_processed_dir)
os.mkdir(testing_processed_dir)
os.mkdir(training_label_1_dir)
os.mkdir(training_label_2_dir)
os.mkdir(testing_label_1_dir)
os.mkdir(testing_label_2_dir)

TEST_SPLIT = 0.1
MIN_TEST_SIZE = 5

# CHASE DB
# ------------------------------------------------------------------------------------------------
ChaseDB_dir_path = 'Dataset/CHASEDB/'
chase_files_labels = np.array(glob(ChaseDB_dir_path + '*.png'))
chase_files_input = np.array(glob(ChaseDB_dir_path + '*.jpg'))
test_count = max(int(TEST_SPLIT * len(chase_files_input)), MIN_TEST_SIZE)
index = np.random.permutation(len(chase_files_input))
chase_files_input = chase_files_input[index]
chase_files_labels = chase_files_labels[index]
k = 1
print("TRAINING DIR -----------------")
for i, file in enumerate(chase_files_input[:-test_count], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(training_input_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(file, training_input_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(ChaseDB_dir_path + image_name + '_1stHO.png', label_1_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(ChaseDB_dir_path + image_name + '_2ndHO.png', label_2_dir + '/CHASE_{:02d}.png'.format(i))
    label_1 = Image.open(ChaseDB_dir_path + image_name + '_1stHO.png')
    label_2 = Image.open(ChaseDB_dir_path + image_name + '_2ndHO.png')
    label_1.convert("L").save(training_label_1_dir + '/CHASE_{:02d}.png'.format(i))
    label_2.convert("L").save(training_label_2_dir + '/CHASE_{:02d}.png'.format(i))
    k += 1
print("TESTING DIR -----------------")
for i, file in enumerate(chase_files_input[-test_count:], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(testing_input_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(file, testing_input_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(ChaseDB_dir_path + image_name + '_1stHO.png', label_1_dir + '/CHASE_{:02d}.png'.format(i))
    # shutil.copy(ChaseDB_dir_path + image_name + '_2ndHO.png', label_2_dir + '/CHASE_{:02d}.png'.format(i))
    label_1 = Image.open(ChaseDB_dir_path + image_name + '_1stHO.png')
    label_2 = Image.open(ChaseDB_dir_path + image_name + '_2ndHO.png')
    label_1.convert("L").save(testing_label_1_dir + '/CHASE_{:02d}.png'.format(i))
    label_2.convert("L").save(testing_label_2_dir + '/CHASE_{:02d}.png'.format(i))

# STARE
# ------------------------------------------------------------------------------------------------
Stare_dir_path = 'Dataset/STARE/training-set/'
Stare_label_1_dir = 'Dataset/STARE/labels-ah-jpg/'
Stare_label_2_dir = 'Dataset/STARE/labels-vk-jpg/'
stare_files = np.array(glob(Stare_dir_path + '*.jpg'))

test_count = max(int(TEST_SPLIT * len(stare_files)), MIN_TEST_SIZE)
index = np.random.permutation(len(stare_files))
stare_files = stare_files[index]
k = 1
print("TRAINING DIR -----------------")
for i, file in enumerate(stare_files[:-test_count], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(training_input_dir + '/STARE_{:02d}.png'.format(i))
    # shutil.copy(file, training_input_dir + '/STARE_{:02d}.png'.format(i))
    # shutil.copy(Stare_label_1_dir + image_name + 'ah.jpg', label_1_dir + '/STARE_{:02d}.jpg'.format(i))
    # shutil.copy(Stare_label_2_dir + image_name + 'vk.jpg', label_2_dir + '/STARE_{:02d}.jpg'.format(i))
    label_1 = Image.open(Stare_label_1_dir + image_name + 'ah.jpg')
    label_2 = Image.open(Stare_label_2_dir + image_name + 'vk.jpg')
    label_1.convert('L').save(training_label_1_dir + '/STARE_{:02d}.png'.format(i))
    label_2.convert('L').save(training_label_2_dir + '/STARE_{:02d}.png'.format(i))
    k += 1
print("TESTING DIR -----------------")
for i, file in enumerate(stare_files[-test_count:], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(testing_input_dir + '/STARE_{:02d}.png'.format(i))
    # shutil.copy(file, testing_input_dir + '/STARE_{:02d}.png'.format(i))
    # shutil.copy(Stare_label_1_dir + image_name + 'ah.jpg', label_1_dir + '/STARE_{:02d}.jpg'.format(i))
    # shutil.copy(Stare_label_2_dir + image_name + 'vk.jpg', label_2_dir + '/STARE_{:02d}.jpg'.format(i))
    label_1 = Image.open(Stare_label_1_dir + image_name + 'ah.jpg')
    label_2 = Image.open(Stare_label_2_dir + image_name + 'vk.jpg')
    label_1.convert('L').save(testing_label_1_dir + '/STARE_{:02d}.png'.format(i))
    label_2.convert('L').save(testing_label_2_dir + '/STARE_{:02d}.png'.format(i))


# DRIVE
# ------------------------------------------------------------------------------------------------
drive_dir = 'Dataset/DRIVE/images/'
drive_label_dir = 'Dataset/DRIVE/1st_manual/'
drive_files = np.array(glob(drive_dir + '*.tif'))

test_count = max(int(TEST_SPLIT * len(drive_files)), MIN_TEST_SIZE)
index = np.random.permutation(len(drive_files))
drive_files = drive_files[index]
k = 1
print("TRAINING DIR -----------------")
for i, file in enumerate(drive_files[:-test_count], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(training_input_dir + '/DRIVE_{:02d}.png'.format(i))
    # shutil.copy(file, training_input_dir + '/DRIVE_{:02d}.png'.format(i))
    # shutil.copy(drive_training_label_dir + image_name.replace('training', 'manual1') + '.gif', label_1_dir + '/DRIVE_{:02d}.gif'.format(i))
    label_1 = Image.open(drive_label_dir + image_name + '.gif')
    label_1.convert('L').save(training_label_1_dir + '/DRIVE_{:02d}.png'.format(i))
    k += 1
print("TESTING DIR -----------------")
for i, file in enumerate(drive_files[-test_count:], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(testing_input_dir + '/DRIVE_{:02d}.png'.format(i))
    # shutil.copy(file, testing_input_dir + '/DRIVE_{:02d}.png'.format(i))
    # shutil.copy(drive_training_label_dir + image_name.replace('training', 'manual1') + '.gif', label_1_dir + '/DRIVE_{:02d}.gif'.format(i))
    label_1 = Image.open(drive_label_dir + image_name + '.gif')
    label_1.convert('L').save(testing_label_1_dir + '/DRIVE_{:02d}.png'.format(i))


# HRF
# ------------------------------------------------------------------------------------------------
hrf_dir = 'Dataset/HRF/images/'
hrf_label_dir = 'Dataset/HRF/manual1/'
hrf_files = np.array(glob(hrf_dir + '*.jpg'))

test_count = max(int(TEST_SPLIT * len(hrf_files)), MIN_TEST_SIZE)
index = np.random.permutation(len(hrf_files))
hrf_files = hrf_files[index]
k = 1
print("TRAINING DIR -----------------")
for i, file in enumerate(hrf_files[:-test_count], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(training_input_dir + '/HRF_{:02d}.png'.format(i))
    # shutil.copy(file, training_input_dir + '/DRIVE_{:02d}.png'.format(i))
    # shutil.copy(drive_training_label_dir + image_name.replace('training', 'manual1') + '.gif', label_1_dir + '/DRIVE_{:02d}.gif'.format(i))
    label_1 = Image.open(hrf_label_dir + image_name + '.tif')
    label_1.convert('L').save(training_label_1_dir + '/HRF_{:02d}.png'.format(i))
    k += 1
print("TESTING DIR -----------------")
for i, file in enumerate(hrf_files[-test_count:], k):
    print(file)
    image_name = ''.join(file.replace('\\', '/').split('/')[-1].split('.')[:-1])
    Image.open(file).save(testing_input_dir + '/HRF_{:02d}.png'.format(i))
    # shutil.copy(file, testing_input_dir + '/DRIVE_{:02d}.png'.format(i))
    # shutil.copy(drive_training_label_dir + image_name.replace('training', 'manual1') + '.gif', label_1_dir + '/DRIVE_{:02d}.gif'.format(i))
    label_1 = Image.open(hrf_label_dir + image_name + '.tif')
    label_1.convert('L').save(testing_label_1_dir + '/HRF_{:02d}.png'.format(i))
