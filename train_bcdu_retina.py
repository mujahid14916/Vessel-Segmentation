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
TOTAL_BATCHES = 10000
TOTAL_VAL_DATA_BATCHES = 1000
WEIGHT_FILE_NAME = 'models/bcdu_weight_lstm.hdf5'
EPOCHS = 5

#model = M.unet2_segment(input_size = (64,64,1))
print("Initializing Network")
model = M.BCDU_net_D3(input_size = (*PATCH_SIZE, 1))
if os.path.isfile(WEIGHT_FILE_NAME):
    print("Loading Saved Model")
    model.load_weights(WEIGHT_FILE_NAME)
model.summary()


# mcp_save = ModelCheckpoint('weight_lstm.hdf5', save_best_only=True, monitor='val_loss', mode='min')
mcp_save = ModelCheckpoint('models/bcdu_weight-{epoch:02d}-{val_accuracy:.6f}.hdf5', monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')

history = model.fit_generator(data_generator('training_dataset', 
                                             'pre-processed', 
                                             'label-1', 'png', 
                                             batch_size=BATCH_SIZE, 
                                             patch_size=PATCH_SIZE),
                              steps_per_epoch=TOTAL_BATCHES,
                              epochs=EPOCHS,
                              validation_data=data_generator('testing_dataset', 
                                                             'pre-processed', 
                                                             'label-1', 
                                                             'png', 
                                                             batch_size=BATCH_SIZE, 
                                                             patch_size=PATCH_SIZE),
                              validation_steps=TOTAL_VAL_DATA_BATCHES,
                              callbacks=[mcp_save, reduce_lr_loss],
                              class_weight=[1, 5])  # TODO: Check Class weights

with open('training_history/bcdu_{}.out'.format(time()), 'wb') as f:
    d = {}
    d['epoch'] = history.epoch
    d['history'] = history.history
    pickle.dump(d, f)
