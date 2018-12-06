import datetime
import getpass
import os

import keras
import keras.backend as K
import tensorflow as tf

import image_util

tf.logging.set_verbosity(tf.logging.INFO)


class Wrapper:

    def __init__(self, model, optimizer, loss_function, batch_size, num_epochs, learning_rate):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self.image_batch, self.label_batch = self.inputs(batch_size=batch_size, num_epochs=num_epochs)

        if loss_function == 'rmse':
            self.loss_function = self.root_mean_squared_error

        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss_function,
                           metrics=['accuracy'])

    def decode(self, serealised):
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

    def inputs(self, batch_size, num_epochs):
        with tf.name_scope('input'):
            dataset = tf.data.TFRecordDataset(image_util.train_tensor_file)
            dataset = dataset.map(self.decode)
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.batch(batch_size, drop_remainder=True)

            iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def root_mean_squared_error(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def run(self):
        log_dir = './logs/{0:%y%m%d_%H%M%S}_{1}/'.format(datetime.datetime.now(), getpass.getuser())
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("The directory for this run is {}".format(log_dir))
        print("Don't forget to make notes on the performance of the model in the notes.txt file!")
        print("You can run tensorboard with:")
        print("  tensorboard --logdir=/home/george/Documents/4th_year/neural/logs")

        with open(log_dir + "model.json", 'w+') as f:
            f.write(self.model.to_json())

        with open(log_dir + "notes.txt", 'w+') as f:
            f.write("optimizer: {}".format(self.optimizer))
            f.write("loss function: {}".format(self.loss_function))
            f.write("batch size: {}".format(self.batch_size))
            f.write("# Epochs: {}".format(self.num_epochs))
            f.write("Learning rate: {}".format(self.learning_rate))

        tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir,
                                                  histogram_freq=0,
                                                  batch_size=self.batch_size,
                                                  write_graph=True,
                                                  write_grads=False,
                                                  write_images=False,
                                                  embeddings_freq=0,
                                                  embeddings_layer_names=None,
                                                  embeddings_metadata=None,
                                                  embeddings_data=None,
                                                  update_freq='epoch')

        self.model.fit(self.image_batch, self.label_batch, epochs=self.num_epochs, steps_per_epoch=1,
                       callbacks=[tensorboard])

        print("The directory for this run is {}".format(log_dir))
        print("Don't forget to make notes on the performance of the model in the notes.txt file!")

    def main(self):
        tf.app.run(main=self.run())
