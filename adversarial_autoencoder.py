import tensorflow as tf
import numpy as np
import datetime
import os
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Progressbar
# bar = progressbar.ProgressBar(widgets=['[', progressbar.Timer(), ']', progressbar.Bar(), '(', progressbar.ETA(), ')'])

# Get the MNIST data
# mnist = input_data.read_data_sets('./Data', one_hot=True)
dataset = np.load("data/training_data.npy")
shuffled_dataset = dataset.copy()

# Parameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
n_l3 = 1000

TINY = 1e-8

IM_SIZE = 64
NB_CHANNELS = 3
Z_DIM = 35
BATCH_SIZE = 256
N_EPOCHS = 500
learning_rate = 0.00005
beta1 = 0.9
TRAIN_DISC_EVERY = 1
results_path = './Results/Adversarial_Autoencoder'

# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IM_SIZE, IM_SIZE, NB_CHANNELS], name='Input')
x_target = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IM_SIZE, IM_SIZE, NB_CHANNELS], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, Z_DIM], name='Real_distribution')
decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, Z_DIM], name='Decoder_input')


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}_{1}_{2}_{3}_{4}_{5}_Adversarial_Autoencoder". \
        format(datetime.datetime.now(), Z_DIM, learning_rate, BATCH_SIZE, N_EPOCHS, beta1)
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    x_points = np.arange(-10, 10, 1.5).astype(np.float32)
    y_points = np.arange(-10, 10, 1.5).astype(np.float32)

    nx, ny = len(x_points), len(y_points)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)

    for i, g in enumerate(gs):
        z = np.concatenate(([x_points[int(i / ny)]], [y_points[int(i % nx)]]))
        z = np.reshape(z, (1, 2))
        x = sess.run(op, feed_dict={decoder_input: z})
        ax = plt.subplot(g)
        img = np.array(x.tolist()).reshape(28, 28)
        ax.imshow(img, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('auto')
    plt.show()


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [BATCH_SIZE, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out


# The autoencoder network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.elu(dense(x, input_dim, n_l1, 'e_dense_1'))
        e_dense_2 = tf.nn.elu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        latent_variable = dense(e_dense_2, n_l2, Z_DIM, 'e_latent_variable')
        return latent_variable


def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.elu(dense(x, Z_DIM, n_l2, 'd_dense_1'))
        d_dense_2 = tf.nn.elu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output


# The autoencoder network
def encoder2d(x, reuse=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :return: tensor which is the hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        conv1 = tf.layers.conv2d(
            inputs=x,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.elu,
            name="e_conv1")
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, name='e_pool1')
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.elu,
            name="e_conv2")
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name='e_pool2')
        # pool2_flattened = tf.layers.flatten(pool2, name='e_pool2flatten')

        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.elu,
            name="e_conv3")
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2, name='e_pool3')
        pool3_flattened = tf.layers.flatten(pool3, name='e_pool3flatten')

        batch_norm1 = tf.layers.batch_normalization(pool3_flattened, name='e_batch_norm1')
        fc1 = tf.layers.dense(inputs=batch_norm1, units=1024, activation=tf.nn.elu, name='e_fc1')
        batch_norm2 = tf.layers.batch_normalization(fc1, name='e_batch_norm2')
        #fc2 = tf.layers.dense(inputs=batch_norm2, units=Z_DIM, activation=tf.nn.elu, name='e_fc2')
        fc2 = dense(batch_norm2, 1024, Z_DIM, 'e_fc2')

        print fc2.get_shape()
        return fc2


def decoder2d(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_fc1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.elu, name='d_fc1')
        d_batch_norm1 = tf.layers.batch_normalization(d_fc1, name='d_batch_norm1')
        d_fc2 = tf.layers.dense(inputs=d_batch_norm1, units=int(IM_SIZE/8*IM_SIZE/8*128), activation=tf.nn.elu, name='d_fc2')
        d_batch_norm2 = tf.layers.batch_normalization(d_fc2, name='d_batch_norm2')
        d_fc2_reshaped = tf.reshape(d_batch_norm2, [-1, int(IM_SIZE/8), int(IM_SIZE/8), 128], name='d_fc2flatten')

        deconv1 = tf.layers.conv2d(
            inputs=d_fc2_reshaped,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.elu,
            name="d_conv1")
        upsample1 = tf.image.resize_images(
            images=deconv1,
            size=[IM_SIZE / 4, IM_SIZE / 4],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        deconv2 = tf.layers.conv2d(
                                   inputs=upsample1,
                                   filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation=tf.nn.elu,
                                   name="d_conv2")
        upsample2 = tf.image.resize_images(
                                           images=deconv2,
                                           size=[IM_SIZE / 2, IM_SIZE / 2],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        deconv3 = tf.layers.conv2d(
                                   inputs=upsample2,
                                   filters=64,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation=tf.nn.elu,
                                   name="d_conv3")
        upsample3 = tf.image.resize_images(
                                           images=deconv3,
                                           size=[IM_SIZE, IM_SIZE],
                                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # deconv4 = tf.layers.conv2d(
        #                            inputs=upsample3,
        #                            filters=32,
        #                            kernel_size=[3, 3],
        #                            padding="same",
        #                            activation=tf.nn.elu,
        #                            name="d_conv4")
        # upsample4 = tf.image.resize_images(
        #                                    images=deconv4,
        #                                    size=[IM_SIZE, IM_SIZE],
        #                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        decoded = tf.layers.conv2d(
                                   inputs=upsample3,
                                   filters=NB_CHANNELS,
                                   kernel_size=[3, 3],
                                   padding="same",
                                   activation=tf.nn.sigmoid,
                                   name='d_lastconv')
        return decoded


def discriminator(x, reuse=False, normalize=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [BATCH_SIZE, Z_DIM]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [BATCH_SIZE, 1]
    """

    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        if normalize:
            mean, var = tf.nn.moments(x, axes=[1], name='dc_meanvar', keep_dims=True)
            x_begin = tf.div(tf.subtract(x, mean), tf.sqrt(var), name='dc_xbegin')
        else:
            x_begin = x
        # dc_den1 = tf.nn.elu(dense(x_begin, Z_DIM, n_l1, name='dc_den1'))
        # dc_den2 = tf.nn.elu(dense(dc_den1, n_l1, n_l2, name='dc_den2'))
        # dc_batch_norm1 = tf.layers.batch_normalization(dc_den2, name='dc_batch_norm1')
        #
        # output = dense(dc_batch_norm1, n_l2, 1, name='dc_output')

        # Version with 3 FC since it seems discriminator isn't good enough
        dc_den1 = tf.nn.elu(dense(x_begin, Z_DIM, n_l1, name='dc_den1'))
        dc_den2 = tf.nn.elu(dense(dc_den1, n_l1, n_l2, name='dc_den2'))
        dc_den3 = tf.nn.elu(dense(dc_den2, n_l2, n_l3, name='dc_den3'))
        dc_batch_norm1 = tf.layers.batch_normalization(dc_den3, name='dc_batch_norm1')

        output = dense(dc_batch_norm1, n_l2, 1, name='dc_output')
        return output


def train(train_model=True):
    """
    Used to train the autoencoder by passing in the necessary inputs.
    :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
    :return: does not return anything
    """
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder2d(x_input)
        decoder_output = decoder2d(encoder_output)
        print decoder_output.get_shape()

    with tf.variable_scope(tf.get_variable_scope()):
        d_real = discriminator(real_distribution)
        d_fake = discriminator(encoder_output, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder2d(decoder_input, reuse=True)

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Discrimminator Loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    # # LOSSES AS EXPLAINED IN PAPER
    # sigm_real = tf.nn.sigmoid(d_real)
    # sigm_fake = tf.nn.sigmoid(d_fake)
    #
    # # Discrimminator Loss
    # dc_loss = -tf.reduce_mean(tf.log(sigm_real + TINY) + tf.log(1. - sigm_fake + TINY))
    #
    # # Generator loss
    # generator_loss = -tf.reduce_mean(tf.log(sigm_fake + TINY))

    all_variables = tf.trainable_variables()
    dc_var = [var for var in all_variables if 'dc_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(dc_loss, var_list=dc_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)

    init = tf.global_variables_initializer()

    # Reshape immages to display them
    input_images = tf.reshape(x_input, [-1, IM_SIZE, IM_SIZE, NB_CHANNELS])
    generated_images = tf.reshape(decoder_output, [-1, IM_SIZE, IM_SIZE, NB_CHANNELS])

    # Tensorboard visualization
    tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    tf.summary.histogram(name='Real Distribution', values=real_distribution)
    tf.summary.image(name='Input Images', tensor=input_images, max_outputs=3)
    tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=3)
    summary_op = tf.summary.merge_all()

    # Saving the model
    saver = tf.train.Saver()
    step = 0

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        if train_model:
            tensorboard_path, saved_model_path, log_path = form_results()
            sess.run(init)
            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
            for i in range(N_EPOCHS):
                n_batches = int(len(dataset) / BATCH_SIZE)
                print("------------------Epoch {}/{}------------------".format(i, N_EPOCHS))
                np.random.shuffle(shuffled_dataset)
                for b in range(1, n_batches + 1):
                    z_real_dist = np.random.randn(BATCH_SIZE, Z_DIM) * 5.
                    # batch_x, _ = mnist.train.next_batch(BATCH_SIZE)

                    begin = (b-1)*BATCH_SIZE
                    end = min(b*BATCH_SIZE, len(dataset))
                    batch_x = shuffled_dataset[begin:end]

                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    if b % TRAIN_DISC_EVERY == 0:
                        sess.run(discriminator_optimizer,
                                 feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})

                    if b % 5 == 0:
                        a_loss, d_loss, g_loss, summary = sess.run(
                            [autoencoder_loss, dc_loss, generator_loss, summary_op],
                            feed_dict={x_input: batch_x, x_target: batch_x,
                                       real_distribution: z_real_dist})
                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {}, iteration: {}".format(i, b))
                        print("Autoencoder Loss: {}".format(a_loss))
                        print("Discriminator Loss: {}".format(d_loss))
                        print("Generator Loss: {}".format(g_loss))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Autoencoder Loss: {}\n".format(a_loss))
                            log.write("Discriminator Loss: {}\n".format(d_loss))
                            log.write("Generator Loss: {}\n".format(g_loss))
                    step += 1

                saver.save(sess, save_path=saved_model_path, global_step=step)
        else:
            # Get the latest results folder
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' + all_results[-1] + '/Saved_models/'))
            generate_image_grid(sess, op=decoder_image)

if __name__ == '__main__':
    train(train_model=True)
