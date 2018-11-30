from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
from datetime import datetime
import os
import random
import sys
import threading

import numpy as np
import tensorflow as tf
from PIL import Image

root_dir = "/home/george/Documents/4th_year/neural/"
data_dir = root_dir + "data/"
train_dir = data_dir + "train/"
annotations_file = train_dir + "annotations.csv"
train_tensor_file = data_dir + "train_tensors.tfrecords"


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_images(count=-1, output=0, filt=lambda i, lb, lv: True):
    # Read the annotations file
    images = []
    labels = []
    levels = []
    with open(annotations_file) as f:
        reader = csv.reader(f, delimiter=',')
        i = 1
        for row in reader:
            if i == count:
                break
            image_file = "{}{}.png".format(train_dir, row[0])
            img = np.array(Image.open(image_file))
            # images.append(img)
            # labels.append(row[1:])
            # levels.append(int(row[0][0]))
            labels, level = list(map(int, row[1:])), int(row[0][0])
            if filt(img, labels, level):
                yield img, labels, level
            if output != 0:
                if i % output == 0:
                    tf.logging.info("Loaded {} entries".format(i))
            i += 1


def write_tensor(count=-1, output=0, filt=lambda i, lb, lv: True):
    tf.logging.info("Writing tensor...")
    i = 0
    with tf.python_io.TFRecordWriter(train_tensor_file) as writer:
        for img, label, level in load_images(count=count, output=output, filt=filt):
            image_raw = img.tostring()
            tensor = tf.train.Example(features=tf.train.Features(feature={
                'height': _int64_feature(img.shape[0]),
                'width': _int64_feature(img.shape[1]),
                'level': _int64_feature(level),
                'x1': _int64_feature(label[0]),
                'y1': _int64_feature(label[1]),
                'x2': _int64_feature(label[2]),
                'y2': _int64_feature(label[3]),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(tensor.SerializeToString())
            if output != 0:
                if i % output == 0:
                    tf.logging.info("Written entry {}".format(i))
            i += 1

def mean(count=-1, output=0, filt=lambda i, lb, lv: True):
    imgs = load_images(count=count, output=output, filt=filt)
    m, _, _ = imgs.__next__()
    i = 1
    for img, _, _ in imgs:
        m += img
        i += 1
    return m / i

def variance(count=-1, output=0, filt=lambda i, lb, lv: True):
    imgs = load_images(count=count, output=output, filt=filt)
    m, _, _ = imgs.__next__()
    v = m ** 2
    i = 1
    for img, _, _ in imgs:
        v += img ** 2
        m += img
        i += 1
    return (v / i) - ((m / i) ** 2)