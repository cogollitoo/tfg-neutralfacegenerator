from __future__ import division, print_function, absolute_import
from tensorflow.python.util import deprecation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2 as cv2
import glob as glob
import keras as ker
from keras import backend as K
from tools import encoder,decoder,print_info,file_to_print,latent_space,vae_loss

deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.contrib._warning = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

image_dim = 128
# Training Parameters
learning_rate = 1e-3
batch_size = 4
epochs = 2000
# Testing Parameters
samples = 16
reconstruction_images = 4
latent_dim = 256
activation = tf.nn.relu
dirName = '/home/jacostah/tfg/Models_log/' + 'face_VAE' + activation.__str__() \
          + '_learning_rate_' + learning_rate.__str__() \
          + '_with_latent_' + latent_dim.__str__() \
          + '_epochs_' + epochs.__str__()

imgDataset=[]
filenames = [img for img in glob.glob("/home/jacostah/tfg/aligned/*.png")]
filenames.sort()
images = [cv2.imread(img)/255. for img in filenames]

print(len(images[0]))
print(len(images[0][0]))
print(len(images[0][0][0]))

# def print_images():
#     for i in range(reconstruction_images):
#         cv2.imwrite(dirName+'/original_img'+i.__str__()+'.png', images[i]*255.)