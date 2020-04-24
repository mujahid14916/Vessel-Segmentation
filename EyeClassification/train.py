import tensorflow as tf
from matplotlib import pyplot as plt
import os
from glob import glob
import numpy as np

image_size = (256, 256)
Input_Shape = (*image_size, 3)
batch_size = 16
training_data_dir = 'dataset/training'
RES_DIR = 'new_model_2'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)
epochs = 800

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    # brightness_range=(0.1, 0.5),
    channel_shift_range=50,
    rescale=1/255.,
    # vertical_flip=True,
    width_shift_range=0.07,
    height_shift_range=0.07,
    fill_mode='constant',
    zoom_range=[0.8, 1.2]
)

train_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='training'
)

od_images = glob('dataset/testing/OD/*.png')
val_x = []
val_y = []
for file in od_images:
    image = np.asarray(tf.keras.preprocessing.image.load_img(file, target_size=Input_Shape[:2]))
    if np.max(image) > 1:
        image = image/255.
    val_x.append(image)
    val_y.append(0)
os_images = glob('dataset/testing/OS/*.png')
for file in os_images:
    image = np.asarray(tf.keras.preprocessing.image.load_img(file, target_size=Input_Shape[:2]))
    if np.max(image) > 1:
        image = image/255.
    val_x.append(image)
    val_y.append(1)
val_x = np.array(val_x)
val_y = np.array(val_y)

input_layer = tf.keras.layers.Input(Input_Shape)

x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal')(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Conv2D(filters=512, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer = 'he_normal')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid')(x)
    # keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu'),
    # keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu'),
    # keras.layers.MaxPool2D(pool_size=(2, 2)),
    # keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='elu'),
    # keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='elu'),
    # keras.layers.MaxPool2D(pool_size=(2, 2)),
    # keras.layers.MaxPool2D(pool_size=(2, 2)),
    # keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal'),
    # keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
x = tf.keras.layers.Flatten()(x)
    # keras.layers.Dropout(rate=0.5),
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(rate=0.1)(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=input_layer, outputs=output)

# model.load_weights('weights-1723-0.0019-0.9987-0.0436-0.9451.hdf5')

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)
save_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=RES_DIR + '/weights-{epoch:04d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', 
    monitor='val_accuracy', 
    save_best_only=True,
    period=1
)

# model.fit(
#     x=train_generator,
#     epochs=epochs,
#     callbacks=[save_model_callback],
# )

# if not os.path.isdir(RES_DIR):
#     os.mkdir(RES_DIR)
# k = 0
# for i in train_generator:
#     images = i[0]
#     for image in images:
#         print(np.max(image))
#         tf.keras.preprocessing.image.save_img('{}/{:05d}.jpg'.format(RES_DIR, k), image)
#         k += 1
#     if k > 100:
#         break
# exit()

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[save_model_callback],
    validation_data=[val_x, val_y]
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('{}/train_acc-new.png'.format(RES_DIR), dpi=300)
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('{}/train_loss-new.png'.format(RES_DIR), dpi=300)