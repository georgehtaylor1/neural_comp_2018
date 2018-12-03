from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

import image_util

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
        dataset = dataset.shuffle(image_util.train_image_count)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def run_trainin(model):
    with tf.Graph().as_default():
        image_batch, label_batch = inputs(
            train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)


def main(_):
    sess = tf.Session()
    tf.keras.backend.set_session(sess)
    loss = lambda gs, ts: tf.sqrt(tf.reduce_mean(tf.squared_difference(gs, ts)))

    image_batch, label_batch = inputs(batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10000, input_shape=(250000,), activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(100, activation=tf.nn.sigmoid),
        tf.keras.layers.Dense(4)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam,
                  loss='mse',
                  metrics=['accuracy'])

    model.fit(image_batch, label_batch, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size)


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
        default=2,
        help='Number of epochs to run trainer.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Batch size.')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run()
