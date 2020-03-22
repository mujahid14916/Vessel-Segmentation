from SegCaps.capsnet import CapsNetR3, CapsNetR4
from glob import glob
import tensorflow as tf
import numpy as np
from patch_generator import data_generator, full_image_generator
import os
from SegCaps.custom_losses import weighted_binary_crossentropy_loss


PATCH_SIZE = (256, 256)
BATCH_SIZE = 1
INPUT_SHAPE = (*PATCH_SIZE, 3)
SAVED_MODEL_PATH = 'models/segcaps-full-model-100-0.220064-0.884914.hdf5'
INITIAL_EPOCH = 0
EPOCHS = 10


def main():
    model_list = CapsNetR3(INPUT_SHAPE)
    train_model = model_list[0]
    train_model.summary()

    validation_X = []
    validation_Y = []
    files = glob('testing_dataset/patches/*.png')
    for i, file in enumerate(files):
        image = np.asarray(tf.keras.preprocessing.image.load_img(file, target_size=PATCH_SIZE))/255.
        if i % 2 == 0:
            validation_X.append(image)
        else:
            validation_Y.append(image[:, :, :1])
    validation_X = np.array(validation_X)
    validation_Y = np.array(validation_Y)
    mask = validation_X[:, :, :, 1:2] * validation_Y

    log_dir = 'segcaps_logs_multi_channel'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    mcp_save = tf.keras.callbacks.ModelCheckpoint('models/segcaps-multi-channel-model-{epoch:02d}-{loss:.6f}-{out_seg_accuracy:0.6f}.hdf5', monitor='loss', mode='min')

    train_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, loss={'out_seg': weighted_binary_crossentropy_loss(4), 'out_recon': 'mse'}, metrics=['accuracy'])
    if os.path.isfile(SAVED_MODEL_PATH):
        try:
            train_model.load_weights(SAVED_MODEL_PATH)
            print("Weights Loaded Successfully")
        except:
            print("Failed to load Weights")

    history = train_model.fit(data_generator('training_dataset', 
                              'pre-processed', 
                              'label-1', 'png', 
                              batch_size=BATCH_SIZE, 
                              patch_size=PATCH_SIZE,
                              input_channel=3, caps=True),
                        steps_per_epoch=5000,
                        epochs=EPOCHS,
                        validation_data=[[validation_X, validation_Y], [validation_Y, mask]],
                        callbacks=[tensorboard_callback, mcp_save],
                        initial_epoch=INITIAL_EPOCH)


if __name__ == '__main__':
    main()
