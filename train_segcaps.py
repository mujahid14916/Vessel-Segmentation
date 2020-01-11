'''
Capsules for Object Segmentation (SegCaps)
Original Paper by Rodney LaLonde and Ulas Bagci (https://arxiv.org/abs/1804.04241)
Code written by: Rodney LaLonde
If you use significant portions of this code or the ideas from our paper, please cite it :)
If you have any questions, please email me at lalonde@knights.ucf.edu.

This file is used for training models. Please see the README for details about training.
'''

from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from os.path import join
import numpy as np

from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau, TensorBoard
import tensorflow as tf
from SegCaps.model_helper import create_model

from SegCaps.custom_losses import dice_hard, weighted_binary_crossentropy_loss, dice_loss, margin_loss
from gen_data import data_generator
import pickle
from time import time

def get_loss(net, recon_wei, choice):
    if choice == 'bce':
        loss = 'binary_crossentropy'
    elif choice == 'dice':
        loss = dice_loss
    elif choice == 'mar':
        loss = margin_loss(margin=0.4, downweight=0.5, pos_weight=1.0)
    else:
        raise Exception("Unknow loss_type")

    if net.find('caps') != -1:
        # return {'out_seg': loss, 'out_recon': 'mse'}, {'out_seg': 1., 'out_recon': recon_wei}
        return {'out_seg': loss}, {'out_seg': 1., 'out_recon': recon_wei}
    else:
        return loss, None

def get_callbacks(arguments):
    if arguments.net.find('caps') != -1:
        monitor_name = 'val_out_seg_dice_hard'
    else:
        monitor_name = 'val_dice_hard'

    csv_logger = CSVLogger(join(arguments.log_dir, arguments.output_name + '_log_' + arguments.time + '.csv'), separator=',')
    tb = TensorBoard(arguments.tf_log_dir, batch_size=arguments.batch_size, histogram_freq=0)
    model_checkpoint = ModelCheckpoint(join(arguments.check_dir, arguments.output_name + '_model_' + arguments.time + '.hdf5'),
                                       monitor=monitor_name, save_best_only=True, save_weights_only=True,
                                       verbose=1, mode='max')
    lr_reducer = ReduceLROnPlateau(monitor=monitor_name, factor=0.05, cooldown=0, patience=5,verbose=1, mode='max')
    early_stopper = EarlyStopping(monitor=monitor_name, min_delta=0, patience=25, verbose=0, mode='max')

    return [model_checkpoint, csv_logger, lr_reducer, early_stopper, tb]

def compile_model(net_input_shape, uncomp_model, net='segcaps'):
    # Set optimizer loss and metrics
    opt = Adam(lr=1e-4, beta_1=0.99, beta_2=0.999, decay=1e-6)
    if net.find('caps') != -1:
        metrics = {'out_seg': dice_hard}
    else:
        metrics = [dice_hard]

    loss, loss_weighting = get_loss(net=net,
                                    recon_wei=2, choice='dice')

    uncomp_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return uncomp_model


def plot_training(training_history, arguments):
    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 10))
    f.suptitle(arguments.net, fontsize=18)

    if arguments.net.find('caps') != -1:
        ax1.plot(training_history.history['out_seg_dice_hard'])
        ax1.plot(training_history.history['val_out_seg_dice_hard'])
    else:
        ax1.plot(training_history.history['dice_hard'])
        ax1.plot(training_history.history['val_dice_hard'])
    ax1.set_title('Dice Coefficient')
    ax1.set_ylabel('Dice', fontsize=12)
    ax1.legend(['Train', 'Val'], loc='upper left')
    ax1.set_yticks(np.arange(0, 1.05, 0.05))
    if arguments.net.find('caps') != -1:
        ax1.set_xticks(np.arange(0, len(training_history.history['out_seg_dice_hard'])))
    else:
        ax1.set_xticks(np.arange(0, len(training_history.history['dice_hard'])))
    ax1.grid(True)
    gridlines1 = ax1.get_xgridlines() + ax1.get_ygridlines()
    for line in gridlines1:
        line.set_linestyle('-.')

    ax2.plot(training_history.history['loss'])
    ax2.plot(training_history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.legend(['Train', 'Val'], loc='upper right')
    ax1.set_xticks(np.arange(0, len(training_history.history['loss'])))
    ax2.grid(True)
    gridlines2 = ax2.get_xgridlines() + ax2.get_ygridlines()
    for line in gridlines2:
        line.set_linestyle('-.')

    f.savefig(join(arguments.output_dir, arguments.output_name + '_plots_' + arguments.time + '.png'))
    plt.close()


def train(args, train_list, val_list, u_model, net_input_shape):
    # Compile the loaded model
    model = compile_model(args=args, net_input_shape=net_input_shape, uncomp_model=u_model)
    # Set the callbacks
    callbacks = get_callbacks(args)

    # Training the network
    history = model.fit_generator(
        generate_train_batches(args.data_root_dir, train_list, net_input_shape, net=args.net,
                               batchSize=args.batch_size, numSlices=args.slices, subSampAmt=args.subsamp,
                               stride=args.stride, shuff=args.shuffle_data, aug_data=args.aug_data),
        max_queue_size=40, workers=4, use_multiprocessing=False,
        steps_per_epoch=10000,
        validation_data=generate_val_batches(args.data_root_dir, val_list, net_input_shape, net=args.net,
                                             batchSize=args.batch_size,  numSlices=args.slices, subSampAmt=0,
                                             stride=20, shuff=args.shuffle_data),
        validation_steps=500, # Set validation stride larger to see more of the data.
        epochs=200,
        verbose=1)

    # Plot the training data collected
    plot_training(history, args)



from keras.callbacks import ModelCheckpoint, TensorBoard,ReduceLROnPlateau
PATCH_SIZE = (64, 64)
BATCH_SIZE = 32
TOTAL_BATCHES = 10000
TOTAL_VAL_DATA_BATCHES = 1000
WEIGHT_FILE_NAME = 'models/seg_weight.hdf5'
EPOCHS = 5

net_input_shape = (64, 64, 1)
model_list = create_model(net='segcapsr3', input_shape=net_input_shape)
model_list[0].summary()
model = compile_model(net_input_shape, model_list[0], net='segcaps')

mcp_save = ModelCheckpoint('models/seg_weight-{epoch:02d}-{val_out_seg_accuracy:.6f}.hdf5', monitor='val_loss', mode='min')
history = model.fit_generator(data_generator('training_dataset', 
                                            'pre-processed', 
                                            'label-1', 'png', 
                                            batch_size=BATCH_SIZE, 
                                            patch_size=PATCH_SIZE,
                                            caps=True),
                            steps_per_epoch=TOTAL_BATCHES,
                            epochs=EPOCHS,
                            validation_data=data_generator('testing_dataset', 
                                                            'pre-processed', 
                                                            'label-1', 
                                                            'png', 
                                                            batch_size=BATCH_SIZE, 
                                                            patch_size=PATCH_SIZE,
                                                            caps=True),
                            validation_steps=TOTAL_VAL_DATA_BATCHES,
                            callbacks=[mcp_save])

# print_summary(model=model_list[0], positions=[.38, .65, .75, 1.])

with open('training_history/segcaps_{}.out'.format(time()), 'wb') as f:
    d = {}
    d['epoch'] = history.epoch
    d['history'] = history.history
    pickle.dump(d, f)