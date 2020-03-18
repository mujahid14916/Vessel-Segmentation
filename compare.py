from glob import glob
import numpy as np
import tensorflow as tf


images_to_compare = 20
pattern = 'xxx-th'
combined_pattern = 'xxx-combined-th'
files_list = [
    glob('../retcam/*.png'),
    glob('../retcam_40_bcdu_rotation/*{}.jpg'.format(pattern)),
    glob('../retcam_40_bcdu_rotation/*{}.jpg'.format(combined_pattern)),
    glob('../retcam_caps_results_2_30/*{}.jpg'.format(pattern)),
    glob('../retcam_caps_results_2_30/*{}.jpg'.format(combined_pattern)),
    glob('../retcam_caps_results_2_30-rotation/*{}.jpg'.format(combined_pattern)),
    glob('../retcam_caps_results_full_150/*{}.jpg'.format(combined_pattern)),
]

indexes = np.random.permutation(len(files_list[0]))[:images_to_compare]
files_list[0] = list(map(lambda x: x.replace('.png', 'xxx.png'), files_list[0]))
for files in files_list:
    files.sort()
files_list[0] = list(map(lambda x: x.replace('xxx.png', '.png'), files_list[0]))

image = None
for i in indexes:
    imgs = []
    for files in files_list:
        img = np.asarray(tf.keras.preprocessing.image.load_img(files[i]))
        imgs.append(img)
    print(files_list[0][i])
    print(files_list[1][i])
    print(files_list[2][i])
    if i == indexes[0]:
        image = np.concatenate(imgs, axis=1)
    else:
        image = np.concatenate([image, np.concatenate(imgs, axis=1)], axis=0)

tf.keras.preprocessing.image.save_img('result.png', image)
print(image.shape)
