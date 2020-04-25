from __future__ import division, print_function, absolute_import
from tensorflow.python.util import deprecation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import cv2 as cv2
import glob as glob
from tools import encoder,decoder,print_info,file_to_print,latent_space,vae_loss
import random


deprecation._PRINT_DEPRECATION_WARNINGS = False
tf.contrib._warning = None
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


image_dim = 128
# Training Parameters
learning_rate = 1e-4
batch_size = 19
epochs = 3000
dropout_rate=0.75
# Testing Parameters
samples = 16
reconstruction_images = 4
latent_dim = 256
activation = tf.nn.leaky_relu
dirName = '/home/jacostah/tfg/Models_log/' + 'face_VAE' + activation.__str__() \
          + '_learning_rate_' + learning_rate.__str__() \
          + '_with_latent_' + latent_dim.__str__() \
          + '_epochs_' + epochs.__str__()

imgDataset=[]
filenames = [img for img in glob.glob("/home/jacostah/tfg/aligned/*.png")]
filenames.sort()
images = [cv2.imread(img)/255. for img in filenames]
random.shuffle(images)

def print_images():
    for i in range(reconstruction_images):
        cv2.imwrite(dirName+'/original_img'+i.__str__()+'.png', images[i]*255.)
        cv2.imwrite(dirName + '/reconstructed_img'+i.__str__()+'.png', d[i] * 255.)


##########################################   MODEL   ##########################################
# Construct model
X_true = tf.placeholder("float", [None, image_dim,image_dim,3])
Z_sampled = tf.placeholder("float", [None, latent_dim])
is_training = tf.placeholder(tf.bool,name='is_training')
beta = tf.placeholder(tf.float32, [], name='beta')
d_rate = tf.placeholder_with_default(1.0, shape=(),name='dropout')
# encode
enco = encoder(X_true,activation,image_dim,is_training,d_rate)
# generate distribution
z_mean, z_std = latent_space(enco,latent_dim)
eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
# Sampler: Normal (gaussian) random distribution
# Reparametrization trick
Z_sampled = z_mean + tf.exp(z_std / 2) * eps
# Z_sampled = z_mean + (z_std * eps )
deco = decoder(Z_sampled,activation,is_training,d_rate)
# loss
loss, reconstruction_loss, kdloss = vae_loss(deco, X_true, beta, z_mean, z_std)
# optimizer
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.train.get_global_step(), var_list=tf.global_variables())
    # train_op = optimizer.minimize(loss)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
##########################################   FINISH MODEL   #######################################
# Start Training
# Start a new TF session
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    total_batchs = int(len(images) / batch_size)
    print("Total Batch Size : ", total_batchs)
    print("Total Train Size : ", len(images))
    # TRAINING
    stop = False
    epoch = 1
    err_prev = 0
    count_impr = 0
    count_wrse = 0
    beta_train = 0.0
    f = file_to_print(dirName)
    while epoch < epochs+1 and (not stop):
        # linear
        # beta_train = np.add(np.multiply(0.013, epoch - 1), 0.0001)
        # if epoch > 30:
        #     beta_train = 1.0
        mean = np.zeros(latent_dim, float)
        std = np.zeros(latent_dim, float)
        total_rl = 0.
        total_kdl = 0.
        total_l = 0.
        for i in range(total_batchs):
            batch_x = images[i*batch_size:((i+1)*batch_size)]
            _, l = sess.run([optimizer, loss],
                                           feed_dict={X_true: batch_x, beta: beta_train, is_training: True,d_rate:0.75})
            mean_batch = np.sum(0.0, axis=0) / batch_size
            std_batch = np.sum(0.0, axis=0) / batch_size

            mean += mean_batch
            std += std_batch
            total_rl += 0.0
            total_kdl += 0.0
            total_l += l

        total_rl /= total_batchs
        total_kdl /= total_batchs
        total_l /= total_batchs
        mean /= total_batchs
        std /= total_batchs
        print_info(mean, std, epoch, beta_train, total_rl, total_kdl, total_l,f)
        if(epoch%100 == 0):
            d = sess.run(deco, feed_dict={X_true: images[0:4], beta: 0.0, is_training: False})
            for i in range(reconstruction_images):
                cv2.imwrite(dirName + '/original_img' + i.__str__() + '.png', images[i] * 255.)
                cv2.imwrite(dirName + '/reconstructed_img' + i.__str__()+'_epoch_'+epoch.__str__() + '.png', d[i] * 255.)

        # if total_rl < 20 and count_impr < 6 :
        #     count_impr=0
        #     beta_train+= 0.020
        # else :
        #     count_wrse+=1
        # if err_prev == 0:
        #     err_prev = total_l
        # else:
        #     diff = (err_prev - total_l) / err_prev
        #     print(diff)
        #     if diff > 0.10:
        #         count_wrse=0
        #         count_impr = 0
        #         beta_train += 0.033
        #         err_prev = total_l
        #     elif -0.2 <= diff <= 0.10:
        #         count_impr += 1
        #         if count_impr > 2:
        #             err_prev = total_l
        #             count_impr = 0
        #             count_wrse=0
        #             beta_train += 0.02
        #     elif diff < -0.2:
        #         count_impr=0
        #         count_wrse += 1
        #         if count_wrse > 10:
        #             stop = True
        #
        #     if (total_kdl < 11 or total_rl > 20) and epoch>3 :
        #         break
        epoch += 1

    d = sess.run(deco, feed_dict={X_true: images[0:4], beta: 0.0, is_training: False})

    for i in range(reconstruction_images):
        cv2.imwrite(dirName+'/original_img'+i.__str__()+'.png', images[i]*255.)
        cv2.imwrite(dirName + '/reconstructed_img'+i.__str__()+'.png', d[i] * 255.)

    loss_test = 0.
    loss_test = (loss_test / reconstruction_images)
    f.write('TEST RECONSTRUCTION LOSS : ' + '{:4.3f}'.format(loss_test) + '\n')
