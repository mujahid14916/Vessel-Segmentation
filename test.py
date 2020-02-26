import tensorflow as tf
from glob import glob
import numpy as np

input_size=(480, 640)

ip_files = glob('../Corrected/input/*jpg')
sr_files = glob('../Corrected/raw/*png')
op_files = glob('../Corrected/target/*jpg')
X = []
Y = []
for x, y, z in zip(ip_files, op_files, sr_files):
    x = tf.keras.preprocessing.image.load_img(x, target_size=input_size)
    y = tf.keras.preprocessing.image.load_img(y, target_size=input_size)
    z = tf.keras.preprocessing.image.load_img(z, target_size=input_size)
    x = np.asarray(x)[:, :, :1]/255.
    z = np.asarray(z)[:, :, :1]/255.
    x = np.concatenate([x, z], axis=2)
    X.append(x)
    Y.append(np.asarray(y)[:, :, :1]/255.)

X = np.array(X)
Y = np.array(Y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(*input_size, 1), activation='relu'),
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    # tf.keras.layers.MaxPool2D((2, 2)),
    # tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    # tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu'),
    # tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.UpSampling2D((2, 2)),
    tf.keras.layers.Conv2DTranspose(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid'),
])

k = 7

input_layer = tf.keras.layers.Input(shape=(*input_size, 2))
x = tf.keras.layers.Conv2D(32, (k, k), padding='same', activation='relu')(input_layer)
x1 = tf.keras.layers.MaxPool2D((2, 2))(x)
# -------------------------------------------------------
x = tf.keras.layers.Conv2D(64, (k, k), padding='same', activation='relu')(x1)
x = tf.keras.layers.Conv2D(64, (k, k), padding='same', activation='relu')(x)
x2 = tf.keras.layers.MaxPool2D((2, 2))(x)
# -------------------------------------------------------
x3 = tf.keras.layers.Conv2D(128, (k, k), padding='same', activation='relu')(x2)
x = tf.keras.layers.Conv2D(128, (k, k), padding='same', activation='relu')(x3)
# x = # tf.keras.layers.MaxPool2D((2, 2))(x)
# x = # tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
# x = # tf.keras.layers.Conv2DTranspose(256, (3, 3), padding='same', activation='relu')(x)
# x = # tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(128, (k, k), padding='same', activation='relu')(x)
x = tf.keras.layers.concatenate([x, x3], axis=3)
x = tf.keras.layers.Conv2DTranspose(128, (k, k), padding='same', activation='relu')(x)
# -------------------------------------------------------
x = tf.keras.layers.concatenate([x, x2], axis=3)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(64, (k, k), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(64, (k, k), padding='same', activation='relu')(x)
# -------------------------------------------------------
x = tf.keras.layers.concatenate([x, x1], axis=3)
x = tf.keras.layers.UpSampling2D((2, 2))(x)
x = tf.keras.layers.Conv2DTranspose(32, (k, k), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(1, (k, k), padding='same', activation='sigmoid')(x)



def dice_loss2(y_true, y_pred):
  numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
  denominator = tf.reduce_sum(y_true + y_pred, axis=-1)

  return 1 - (numerator + 1) / (denominator + 1)


model = tf.keras.models.Model(inputs=input_layer, outputs=x)
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, batch_size=4, epochs=500, validation_split=0.2)

y = model.predict(X[10:])
for i, j in enumerate(y):
    tf.keras.preprocessing.image.save_img(f'{i}5.jpg', j)

# from glob import glob

# files = glob('../retcam_caps_results_90/*.jpg')
# manual = glob('../Corrected/target/*.jpg')

# print(files)
# bcdu, segcaps, segcaps_th = [], [], []
# for i, file in enumerate(files):
#     if i % 3 == 0:
#         bcdu.append(file)
#     elif i % 3 == 1:
#         segcaps.append(file)
#     if i % 3 == 2:
#         segcaps_th.append(file)
    
# image_names = []
# for file in bcdu:
#     image_names.append('_'.join(file.replace('\\', '/').split('/')[-1].split('_')[:2]) + '.png')

# from PIL import Image
# import numpy as np
# images = []
# for file in image_names:
#     images.append(np.asarray(Image.open('../retcam/' + file))[:, :, :3])

# bcdu_images = []
# for file in bcdu:
#     bcdu_images.append(np.asarray(Image.open(file))[:, :, :3])
# segcaps_images = []
# for file in segcaps:
#     segcaps_images.append(np.asarray(Image.open(file))[:, :, :3])
# segcaps_th_images = []
# for file in segcaps_th:
#     segcaps_th_images.append(np.asarray(Image.open(file))[:, :, :3])
# manual_images = []
# for file in manual:
#     manual_images.append(np.asarray(Image.open(file))[:, :, :3])

# image = None
# for i in range(len(bcdu)):
#     if image is None:
#         image = np.concatenate([bcdu_images[i], segcaps_images[i], images[i], segcaps_th_images[i], manual_images[i]], axis=1)
#     else:
#         img = np.concatenate([bcdu_images[i], segcaps_images[i], images[i], segcaps_th_images[i], manual_images[i]], axis=1)
#         image = np.concatenate([image, img], axis=0)

# print(image.shape)

# Image.fromarray(image).save('res.jpg')
