from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import keras
import tensorflow as tf

import wrapper

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

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

    model = keras.Sequential()
    model.add(keras.layers.Dense(10, input_shape=(750000,), kernel_initializer='uniform', activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(10000, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(100, activation=tf.nn.sigmoid))
    model.add(keras.layers.Dense(4))

    wrapper = wrapper.Wrapper(model=model,
                              batch_size=FLAGS.batch_size,
                              loss_function='rmse',
                              num_epochs=FLAGS.num_epochs,
                              optimizer='adam',
                              learning_rate=FLAGS.learning_rate)
    wrapper.main()