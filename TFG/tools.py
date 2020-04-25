from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.util import deprecation
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os


# Define VAE Loss
def vae_loss(x_reconstructed, x_true, beta, z_mean, z_std):
    # Reconstruction loss
    x_reconstructed=tf.reshape(x_reconstructed,[-1,128*128*3])
    x_true = tf.reshape(x_true, [-1, 128 * 128 * 3])
    l2_loss = tf.pow((x_reconstructed - x_true),2)
    # l2_loss = (x_true) * tf.log(x_reconstructed+ 1e-2) \
    #                      + (1. - x_true ) * tf.log( 1. - x_reconstructed+ 1e-2)
    # l2_loss = ker.losses.binary_crossentropy(x_true,x_reconstructed)*784
    # l2_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true,logits=x_reconstructed)
    # latent_loss = -0.5 * K.sum(1 + z_std
    #                                    - tf.square(z_mean)
    #                                    - tf.exp(z_std)
    #                                    ,-1)
    # latent_loss = -0.5 * tf.reduce_sum(1 + z_std
    #                                    - tf.square(z_mean)
    #                                    - tf.exp(z_std), 1, name='KL')
    return tf.reduce_mean(l2_loss), 0.0, 0.0


def file_to_print(dirName):
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
        f = open(dirName + "/log.txt", "w+")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    return f


def fc_layer(input, num_outputs, activation, is_training, d_rate):
    layer = tf.layers.dense(input, num_outputs, activation)
    layer = tf.compat.v1.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.dropout(layer, d_rate)
    return layer


def convo_layer(input, nfilters, ksize, activation, is_training, d_rate):
    layer = tf.layers.conv2d(input, nfilters, ksize,strides=(1, 1), activation=activation, padding="SAME")
    layer = tf.compat.v1.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.dropout(layer, d_rate)

    return layer


def unconvo_layer(input, nfilters, ksize, activation, is_training, strides, d_rate):
    layer = tf.layers.conv2d_transpose(input, nfilters, ksize, strides=strides, activation=activation, padding="SAME")
    layer = tf.compat.v1.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.dropout(layer, d_rate)
    return layer


def encoder(input, activation, image_dim, is_training, d_rate=0.0):
    conv0 = convo_layer(input, 32, 3, activation, is_training, d_rate)
    conv1 = convo_layer(conv0, 64, 3, activation, is_training, d_rate)
    conv2 = convo_layer(conv1, 128, 3, activation, is_training, d_rate)
    conv3 = convo_layer(conv2, 256, 3, activation, is_training, d_rate)
    conv4 = convo_layer(conv3, 512, 3, activation, is_training, d_rate)
    # conv5 = convo_layer(conv4, 1024, 3, activation, is_training, d_rate)

    avg = tf.reduce_mean(conv4, [1,2])

    fc0 = fc_layer(avg, 512, activation, is_training, d_rate)
    fc1 = fc_layer(fc0, 256, activation, is_training, d_rate)
    fc2 = fc_layer(fc1, 128, activation, is_training, d_rate)
    return fc2


def latent_space(input, latent_dim):
    mean_ = tf.layers.dense(input, units=latent_dim, name='mean')  # , activation=tf.nn.sigmoid
    std_dev = tf.nn.softplus(tf.layers.dense(input, units=latent_dim), name='std_dev') + 1e-6
    return mean_, std_dev


def decoder(input, activation, is_training, d_rate):
    fc0 = fc_layer(input, 128, activation, is_training, d_rate)
    fc1 = fc_layer(fc0, 256, activation, is_training, d_rate)
    fc2 = fc_layer(fc1, 512, activation, is_training, d_rate)

    rsl = tf.reshape(fc2, shape=[-1, 4, 4, 32])

    # unconv0 = unconvo_layer(rsl, 1024, 3, activation, is_training, (2,2), d_rate)
    unconv1 = unconvo_layer(rsl, 512, 3, activation, is_training, (2,2), d_rate)
    unconv2 = unconvo_layer(unconv1, 256, 3, activation, is_training, (2,2), d_rate)
    unconv3 = unconvo_layer(unconv2, 128, 3, activation, is_training, (2,2), d_rate)
    unconv4 = unconvo_layer(unconv3, 64, 3, activation, is_training, (2,2), d_rate)
    unconv5 = unconvo_layer(unconv4, 32, 3, activation, is_training, (2,2), d_rate)

    reshape = tf.layers.conv2d_transpose(unconv5, 3, 3, strides=(1,1), padding="SAME")
    return reshape


def print_info(mean, std, epoch, beta_train, rl, kdl, l, f):
    info = '-> EPOCH : ' + '{:3.0f}'.format(epoch) + \
           ' , BETA VALUE : ' + '{:4.3f}'.format(beta_train) + \
           ' , RECONSTRUCTION LOSS : ' + '{:4.3f}'.format(rl) + \
           ' , KL LOSS : ' + '{:4.3f}'.format(kdl) + \
           ' , TOTAL LOSS : ' + '{:4.3f}'.format(128*128*3*l) + \
           ' , MEAN : ' + '{:4.3f}'.format(np.mean(mean)) + \
           ' , STD : ' + '{:4.3f}'.format(np.mean(np.exp(std / 2.)))
    print(info)
    f.write(info + '\n')

# ##########################################   MODEL   ##########################################
# # Construct model
# X_true = tf.placeholder("float", [None, image_dim,image_dim,3])
# Z_sampled = tf.placeholder("float", [None, latent_dim])
# is_training = tf.placeholder(tf.bool,name='is_training')
# beta = tf.placeholder(tf.float32, [], name='beta')
# # encode
# enco = encoder(X_true,activation,image_dim,is_training)
# # generate distribution
# z_mean, z_std = latent_space(enco,latent_dim)
# eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0, name='epsilon')
# # Sampler: Normal (gaussian) random distribution
# # Reparametrization trick
# Z_sampled = z_mean + tf.exp(z_std / 2) * eps
# # Z_sampled = z_mean + (z_std * eps )
# deco = decoder(Z_sampled,activation,is_training)
# # loss
# loss, reconstruction_loss, kdloss = vae_loss(deco, X_true, beta, z_mean, z_std)
# # optimizer
# optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=tf.train.get_global_step())
# # Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()
# ##########################################   FINISH MODEL   #######################################
# # Start Training
# # Start a new TF session
# with tf.Session() as sess:
#     # Run the initializer
#     sess.run(init)
#     total_batchs = int(len(images) / batch_size)
#     print("Total Batch Size : ", total_batchs)
#     print("Total Train Size : ", len(images))
#     # TRAINING
#     stop = False
#     epoch = 1
#     err_prev = 0
#     count_impr = 0
#     count_wrse = 0
#     beta_train = 0.0
#     f = file_to_print(dirName)
#     while epoch < epochs+1 and (not stop):
#         # linear
#         # beta_train = np.add(np.multiply(0.013, epoch - 1), 0.0001)
#         # if epoch > 30:
#         #     beta_train = 1.0
#         mean = np.zeros(latent_dim, float)
#         std = np.zeros(latent_dim, float)
#         total_rl = 0.
#         total_kdl = 0.
#         total_l = 0.
#         for i in range(total_batchs):
#             batch_x = images[i*batch_size:((i+1)*batch_size-1)]
#             _, l, rl, kdl, m, s = sess.run([optimizer, loss, reconstruction_loss, kdloss, z_mean, z_std],
#                                            feed_dict={X_true: batch_x, beta: beta_train, is_training: True})
#             mean_batch = np.sum(m, axis=0) / batch_size
#             std_batch = np.sum(s, axis=0) / batch_size
#
#             mean += mean_batch
#             std += std_batch
#             total_rl += rl
#             total_kdl += kdl
#             total_l += l
#
#         total_rl /= total_batchs
#         total_kdl /= total_batchs
#         total_l /= total_batchs
#         mean /= total_batchs
#         std /= total_batchs
#         print_info(mean, std, epoch, beta_train, total_rl, total_kdl, total_l,f)
#
#         # if total_rl < 20 and count_impr < 6 :
#         #     count_impr=0
#         #     beta_train+= 0.020
#         # else :
#         #     count_wrse+=1
#         # if err_prev == 0:
#         #     err_prev = total_l
#         # else:
#         #     diff = (err_prev - total_l) / err_prev
#         #     print(diff)
#         #     if diff > 0.10:
#         #         count_wrse=0
#         #         count_impr = 0
#         #         beta_train += 0.033
#         #         err_prev = total_l
#         #     elif -0.2 <= diff <= 0.10:
#         #         count_impr += 1
#         #         if count_impr > 2:
#         #             err_prev = total_l
#         #             count_impr = 0
#         #             count_wrse=0
#         #             beta_train += 0.02
#         #     elif diff < -0.2:
#         #         count_impr=0
#         #         count_wrse += 1
#         #         if count_wrse > 10:
#         #             stop = True
#         #
#         #     if (total_kdl < 11 or total_rl > 20) and epoch>3 :
#         #         break
#         epoch += 1
#
#     cv2.imshow('original_image', images[0])
#     d = sess.run(deco, feed_dict={X_true: images[0:1], beta: 0.0,is_training:False})
#     cv2.imshow('reconstructed_image', d[0])
#     cv2.imwrite(dirName+'/img1.png', images[0])
#     cv2.waitKey(0)
#
#     loss_test = 0.
#     loss_test = (loss_test / reconstruction_images)
#     f.write('TEST RECONSTRUCTION LOSS : ' + '{:4.3f}'.format(loss_test) + '\n')
