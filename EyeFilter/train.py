import tensorflow as tf
from matplotlib import pyplot as plt
from glob import glob
import numpy as np

image_size = (256, 256)
Input_Shape = (*image_size, 3)
batch_size = 16
training_data_dir = 'dataset/train'
epochs = 200

data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    brightness_range=(0.9, 1.1),
    channel_shift_range=30,
    rescale=1/255.,
)

train_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='training'
)

val_X = []
val_Y = []
for file in glob('dataset/validation/Bad/*.png'):
    img = np.asarray(tf.keras.preprocessing.image.load_img(file, target_size=image_size))
    if np.max(img) > 1:
        img = img / 255.
    val_X.append(img)
    val_Y.append(0)
for file in glob('dataset/validation/Good/*.png'):
    img = np.asarray(tf.keras.preprocessing.image.load_img(file, target_size=image_size))
    if np.max(img) > 1:
        img = img / 255.
    val_X.append(img)
    val_Y.append(1)

val_X = np.array(val_X)
val_Y = np.array(val_Y)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu', input_shape=Input_Shape),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu'),
    tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
    loss=tf.keras.losses.binary_crossentropy,
    metrics=['accuracy']
)
log_dir = 'eye_filter_logs2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
save_model_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='weights-{epoch:04d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', 
    monitor='val_loss', 
    # save_best_only=True,
    period=1
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[save_model_callback, tensorboard_callback],
    validation_data=[val_X, val_Y],
    class_weight={0: 0.8, 1: 0.2}
)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('train_acc.png', dpi=300)
plt.clf()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('train_loss.png', dpi=300)