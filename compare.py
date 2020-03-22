from glob import glob
import numpy as np
import tensorflow as tf


images_to_compare = 20
image_type = 'retcam'
pattern = 'xxx-th'
combined_pattern = 'xxx-combined-th'
files_list = [
    glob('../{}/*.png'.format(image_type)),
    glob('../{}_40_bcdu_rotation/*{}.jpg'.format(image_type, pattern)),
    glob('../{}_40_bcdu_rotation/*{}.jpg'.format(image_type, combined_pattern)),
    glob('../{}_caps_results_2_30/*{}.jpg'.format(image_type, pattern)),
    glob('../{}_caps_results_2_30/*{}.jpg'.format(image_type, combined_pattern)),
    glob('../{}_caps_results_2_30-rotation/*{}.jpg'.format(image_type, combined_pattern)),
    glob('../{}_caps_results_full_150/*{}.jpg'.format(image_type, combined_pattern)),
]

img_size = np.asarray(tf.keras.preprocessing.image.load_img(files_list[1][0])).shape[:2]

indexes = np.random.permutation(len(files_list[0]))[:images_to_compare]
files_list[0] = list(map(lambda x: x.replace('.png', 'xxx.png'), files_list[0]))
for files in files_list:
    files.sort()
files_list[0] = list(map(lambda x: x.replace('xxx.png', '.png'), files_list[0]))

image = None
for i in indexes:
    imgs = []
    for files in files_list:
        img = tf.image.resize(np.asarray(tf.keras.preprocessing.image.load_img(files[i])), img_size)
        imgs.append(img)
    if i == indexes[0]:
        image = np.concatenate(imgs, axis=1)
    else:
        image = np.concatenate([image, np.concatenate(imgs, axis=1)], axis=0)

tf.keras.preprocessing.image.save_img('{}.png'.format(image_type), image)
print(image.shape)
