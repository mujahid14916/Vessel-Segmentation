# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza winchester
"""
import os
import BCDU.models as M
import numpy as np
from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
from keras import callbacks
from time import time
import pickle
from gen_data import data_generator


PATCH_SIZE = (64, 64)
BATCH_SIZE = 64
TOTAL_BATCHES = 1000
TOTAL_VAL_DATA_BATCHES = 100
WEIGHT_FILE_NAME = 'bcdu_weight_lstm.hdf5'
EPOCHS = 10

val_gen = data_generator('testing_dataset', 'pre-processed', 'label-1', 'png', batch_size=BATCH_SIZE, patch_size=PATCH_SIZE)
X_val = None
Y_val = None
for i in range(TOTAL_VAL_DATA_BATCHES):
    x, y = next(val_gen)
    if X_val is None or Y_val is None:
        X_val, Y_val = x, y
    else:
        X_val = np.concatenate([X_val, x], axis=0)
        Y_val = np.concatenate([Y_val, y], axis=0)

#model = M.unet2_segment(input_size = (64,64,1))
model = M.BCDU_net_D3(input_size = (64,64,1))
if os.path.isfile(WEIGHT_FILE_NAME):
    model.load_weights(WEIGHT_FILE_NAME)
model.summary()


# mcp_save = ModelCheckpoint('weight_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='min')
mcp_save = ModelCheckpoint(WEIGHT_FILE_NAME, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit_generator(data_generator('training_dataset', 
                                             'pre-processed', 
                                             'label-1', 'png', 
                                             batch_size=BATCH_SIZE, 
                                             patch_size=PATCH_SIZE),
                              steps_per_epoch=TOTAL_BATCHES,
                              epochs=EPOCHS,
                              validation_data=[X_val, Y_val],
                              callbacks=[mcp_save, reduce_lr_loss])

with open('training_history/bcdu_{}.out'.format(time()), 'wb') as f:
    pickle.dump(history, f)
