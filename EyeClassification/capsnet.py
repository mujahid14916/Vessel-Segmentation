import tensorflow as tf
import keras
from keras.layers import Conv2D, Input, MaxPool2D, Reshape, Lambda
from keras.models import Model
from matplotlib import pyplot as plt
from capsutils import *

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
    class_mode='categorical',
    subset='training'
)

validation_generator = data_generator.flow_from_directory(
    training_data_dir, 
    target_size=image_size, 
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical',
    subset='validation'
)

# A common Conv2D model
input_image = Input(shape=(None, None, 3))
x = Conv2D(64, (3, 3), activation='relu')(input_image)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPool2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
# x = MaxPool2D((2, 2))(x)
# x = Conv2D(256, (3, 3), activation='relu')(x)
# x = Conv2D(256, (3, 3), activation='relu')(x)


"""now we reshape it as (batch_size, input_num_capsule, input_dim_capsule)
then connect a Capsule layer.

the output of final model is the lengths of 10 Capsule, whose dim=16.

the length of Capsule is the proba,
so the problem becomes a 10 two-classification problem.
"""
x = Reshape((-1, 128))(x)
capsule = Capsule(10, 16, 3, True)(x)
# x = Capsule(32, 16, 3, True)(x)
capsule = Capsule(2, 16, 3, True)(capsule)
output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
model = Model(inputs=input_image, outputs=output)

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3), 
    loss=margin_loss,
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