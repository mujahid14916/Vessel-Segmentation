import tensorflow as tf
import keras
from matplotlib import pyplot as plt

image_size = (250, 250)
Input_Shape = (*image_size, 3)
batch_size = 16
training_data_dir = 'dataset'
epochs = 500

data_generator = keras.preprocessing.image.ImageDataGenerator(
    brightness_range=(0.9, 1.1),
    channel_shift_range=30,
    rescale=1/255.,
    validation_split=0.1
)

train_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    class_mode='binary',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal', input_shape=Input_Shape),
    keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal'),
    keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal'),
    keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal'),
    keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal'),
    keras.layers.MaxPool2D(pool_size=(2, 2), padding='valid'),
    # keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer = 'he_normal'),
    # keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4), 
    loss=keras.losses.binary_crossentropy,
    metrics=['accuracy']
)
save_model_callback = keras.callbacks.ModelCheckpoint(
    filepath='weights-{epoch:04d}-{val_loss:.6f}.hdf5', 
    monitor='val_loss', 
    save_best_only=True,
    period=5
)

# model.fit(
#     x=train_generator,
#     epochs=epochs,
#     callbacks=[save_model_callback],
# )

history = model.fit_generator(
    generator=train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    callbacks=[save_model_callback],
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
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