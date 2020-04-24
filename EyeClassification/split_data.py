from glob import glob
import shutil
import os
from random import shuffle

VAL_SPLIT = 0.2
input_dir = 'dataset_full'
od_dir = os.path.join(input_dir, 'OD')
os_dir = os.path.join(input_dir, 'OS')

RES_DIR = 'dataset'
if os.path.isdir(RES_DIR):
    shutil.rmtree(RES_DIR)
training_dir = os.path.join(RES_DIR, 'training')
train_od_dir = os.path.join(training_dir, 'OD')
train_os_dir = os.path.join(training_dir, 'OS')
testing_dir = os.path.join(RES_DIR, 'testing')
test_od_dir = os.path.join(testing_dir, 'OD')
test_os_dir = os.path.join(testing_dir, 'OS')

os.mkdir(RES_DIR)
os.mkdir(training_dir)
os.mkdir(train_od_dir)
os.mkdir(train_os_dir)
os.mkdir(testing_dir)
os.mkdir(test_od_dir)
os.mkdir(test_os_dir)


od_image = glob(os.path.join(od_dir, '*.png'))
os_image = glob(os.path.join(os_dir, '*.png'))
shuffle(od_image)
shuffle(os_image)

os_range = int(len(os_image) * VAL_SPLIT / 2)
od_range = int(len(od_image) * VAL_SPLIT / 2)

for i, file in enumerate(od_image):
    if i > od_range:
        dest = train_od_dir
    else:
        dest = test_od_dir
    shutil.copy(file, dest)

for i, file in enumerate(os_image):
    if i > os_range:
        dest = train_os_dir
    else:
        dest = test_os_dir
    shutil.copy(file, dest)
