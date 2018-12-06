from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
from keras import optimizers
import keras
import keras.backend as K
import image_util
import datetime

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)


def decode(serealised):
    features = tf.parse_single_example(
        serealised,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'level': tf.FixedLenFeature([], tf.int64),
            'x1': tf.FixedLenFeature([], tf.int64),
            'y1': tf.FixedLenFeature([], tf.int64),
            'x2': tf.FixedLenFeature([], tf.int64),
            'y2': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape((250000))

    x1 = tf.cast(features['x1'], tf.int32)
    y1 = tf.cast(features['y1'], tf.int32)
    x2 = tf.cast(features['x2'], tf.int32)
    y2 = tf.cast(features['y2'], tf.int32)
    level = tf.cast(features['level'], tf.int32)
    return image, tf.tuple([x1, y1, x2, y2])


def inputs(batch_size, num_epochs):
    with tf.name_scope('input'):
        dataset = tf.data.TFRecordDataset(image_util.train_tensor_file)
        dataset = dataset.map(decode)
        # dataset = dataset.shuffle(image_util.train_image_count)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size, drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def run_trainin(model):
    with tf.Graph().as_default():
        image_batch, label_batch = inputs(
            train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def main(_):
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    loss = lambda gs, ts: tf.sqrt(tf.reduce_mean(tf.squared_difference(gs, ts)))

    image_batch, label_batch = inputs(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(750000,), kernel_initializer='uniform', activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(10000, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(100, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(4))

    model.compile(optimizer='adam',
                  loss=root_mean_squared_error,
                  metrics=['accuracy'])

    tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/{0:%y%m%d_%H%M%S}/'.format(datetime.datetime.now()),
                                              histogram_freq=0,
                                              batch_size=FLAGS.batch_size,
                                              write_graph=True,
                                              write_grads=False,
                                              write_images=False,
                                              embeddings_freq=0,
                                              embeddings_layer_names=None,
                                              embeddings_metadata=None,
                                              embeddings_data=None,
                                              update_freq='epoch')

    model.fit(image_batch, label_batch, epochs=FLAGS.num_epochs, steps_per_epoch=1, callbacks=[tensorboard])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.')
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=2000,
        help='Number of epochs to run trainer.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='Batch size.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
