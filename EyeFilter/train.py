import tensorflow as tf
import keras
from matplotlib import pyplot as plt

image_size = (256, 256)
Input_Shape = (*image_size, 3)
batch_size = 16
training_data_dir = 'dataset'
epochs = 200

data_generator = keras.preprocessing.image.ImageDataGenerator(
    brightness_range=(0.9, 1.1),
    channel_shift_range=30,
    rescale=1/255.,
    validation_split=0.2
)

train_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary',
    subset='validation'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='elu', input_shape=Input_Shape),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    # keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='elu'),
    keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),
    keras.layers.Flatten(),
    # keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(1, activation='sigmoid'),
])

# model.load_weights('weights-1723-0.0019-0.9987-0.0436-0.9451.hdf5')

model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
    loss=keras.losses.binary_crossentropy,
    metrics=['accuracy']
)
save_model_callback = keras.callbacks.ModelCheckpoint(
    filepath='weights-{epoch:04d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5', 
    monitor='val_accuracy', 
    save_best_only=True,
    period=1
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