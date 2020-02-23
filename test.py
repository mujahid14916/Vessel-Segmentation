# import tensorflow as tf

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(32, (3, 3), input_shape=(512, 512, 1), activation='relu'),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(32, (3, 3), activation='relu'),
#     tf.keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid'),
# ])

# model.summary()

from glob import glob

files = glob('../retcam_caps_results_90/*.jpg')
manual = glob('../Corrected/target/*.jpg')

print(files)
bcdu, segcaps, segcaps_th = [], [], []
for i, file in enumerate(files):
    if i % 3 == 0:
        bcdu.append(file)
    elif i % 3 == 1:
        segcaps.append(file)
    if i % 3 == 2:
        segcaps_th.append(file)
    
image_names = []
for file in bcdu:
    image_names.append('_'.join(file.replace('\\', '/').split('/')[-1].split('_')[:2]) + '.png')

from PIL import Image
import numpy as np
images = []
for file in image_names:
    images.append(np.asarray(Image.open('../retcam/' + file))[:, :, :3])

bcdu_images = []
for file in bcdu:
    bcdu_images.append(np.asarray(Image.open(file))[:, :, :3])
segcaps_images = []
for file in segcaps:
    segcaps_images.append(np.asarray(Image.open(file))[:, :, :3])
segcaps_th_images = []
for file in segcaps_th:
    segcaps_th_images.append(np.asarray(Image.open(file))[:, :, :3])
manual_images = []
for file in manual:
    manual_images.append(np.asarray(Image.open(file))[:, :, :3])

image = None
for i in range(len(bcdu)):
    if image is None:
        image = np.concatenate([bcdu_images[i], segcaps_images[i], images[i], segcaps_th_images[i], manual_images[i]], axis=1)
    else:
        img = np.concatenate([bcdu_images[i], segcaps_images[i], images[i], segcaps_th_images[i], manual_images[i]], axis=1)
        image = np.concatenate([image, img], axis=0)

print(image.shape)

Image.fromarray(image).save('res.jpg')
