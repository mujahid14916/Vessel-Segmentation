import numpy as np
import tensorflow as tf
from glob import glob
from random import shuffle
from matplotlib import pyplot as plt


img_height, img_width = 256, 256


def stich_images(files, row=5, col=5, border_width=0, border_color=(255, 0, 0)):
    image = np.zeros((img_height*row+2*border_width*row, img_width*col+2*border_width*col, 3))
    k = 0
    for i in range(row):
        for j in range(col):
            if k < len(files):
                image[i*img_height+2*i*border_width:i*img_height+2*i*border_width+border_width, 
                      j*img_width+2*j*border_width:(j+1)*img_width+2*j*border_width+border_width, :] = border_color
                image[i*img_height+2*i*border_width:(i+1)*img_height+2*i*border_width+border_width, 
                      j*img_width+2*j*border_width:j*img_width+2*j*border_width+border_width, :] = border_color
                image[(i+1)*img_height+2*i*border_width+border_width:(i+1)*img_height+2*i*border_width+2*border_width, 
                      j*img_width+2*j*border_width:(j+1)*img_width+2*j*border_width+border_width, :] = border_color
                image[i*img_height+2*i*border_width:(i+1)*img_height+2*i*border_width+2*border_width, 
                      (j+1)*img_width+2*j*border_width+border_width:(j+1)*img_width+2*j*border_width+2*border_width, :] = border_color

                # image[i*img_height+2*i*border_width+border_width:(i+1)*img_height+2*i*border_width+border_width, j*img_width+2*j*border_width+border_width:(j+1)*img_width+2*j*border_width+border_width, :] = \
                #                                  np.asarray(tf.keras.preprocessing.image.load_img(files[k]))
                image[i*img_height+2*i*border_width+border_width:(i+1)*img_height+2*i*border_width+border_width, j*img_width+2*j*border_width+border_width:(j+1)*img_width+2*j*border_width+border_width, :] = \
                                                 np.asarray(files[k])
                k += 1
            else:
                break
    return image

#tf.keras.preprocessing.image.save_img('pat.png', stich_images(files, row=4, col=3))
#row, col = 5, 8
#r = 640/480
#img = stich_images(files, row=row, col=col)
#tf.keras.preprocessing.image.save_img('t.png', tf.image.resize(img, (row*240, col*320)))

row, col = 1, 6
files = [
    'input/CHASE_01.png', 
    'label-1/CHASE_01.png',
    'input/DRIVE_01.png', 
    'label-1/DRIVE_01.png',
    'input/STARE_01.png', 
    'label-1/STARE_01.png',
]
images = []

k = 0
temp = []
for i, file in enumerate(files):
    img = np.asarray(tf.keras.preprocessing.image.load_img(file, target_size=(256, 256)))
    
    images.append(img)
img = stich_images(images, row=1, col=6, border_width=0)
tf.keras.preprocessing.image.save_img('res.png', img)
# tf.keras.preprocessing.image.save_img('res.png', img)
