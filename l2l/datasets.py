from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math
import numpy as np
import tensorflow as tf

from cleverhansl2l.dataset import MNIST
from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
from keras.utils import to_categorical

from l2l.util import *

flags = tf.flags
FLAGS = flags.FLAGS


def sample_data(samples, labels):
  """
      Helper method which samples random batches of data

      Args:
          samples: Array of samples
          labels: Array of labels

      Returns:
          A tuple containing a sample of data and the
          corresponding labels
  """
  indices = np.random.randint(
      low=0,
      high=len(samples),
      size=[FLAGS.batch_size])

  batch_samples = np.take(samples, indices, axis=0)
  batch_labels = np.take(labels, indices, axis=0)
  return batch_samples, batch_labels


def batch_data(samples, labels):
  """
      Helper method which splits the dataset in random batches of data

      Args:
          samples: Array of samples
          labels: Array of labels

      Returns:
          A tuple containing the batches of samples & the corresponding
          labels
  """
  pass


def get_mnist(images_shape=get_input_shape_ones(), mode="train"):
  """ Returns the details about MNIST dataset,
      the reshaped images and the labels

  Args:
      images_shape: shape of the images
      mode: select iamges for training or testing

  Returns:
      A tuple where the first element contains meta informarmation
      about the dataset, the second element contains the images
      and the third element contains the labels
  """

  data = mnist_dataset.load_mnist()
  data = getattr(data, mode)

  images = np.reshape(data.images, images_shape)
  labels = to_categorical(data.labels, FLAGS.num_classes)
  return data, images, labels


def sample_mnist(data=None, images=None, labels=None):
  """ Returns a batch of images and
      labels that can be used for training

  Args:
      data: Tensorflow data object containing data attributes
      images: Shaped MNIST Images
      labels: MNIST Labels

  Returns:
      A tuple where the first element is a batch of images
      and the second element is a batch of labels
  """

  if data and images and labels is not None:
    return sample_data(images, labels)
  else:
    data, images, labels = get_mnist()
    return sample_data(images, labels)


def keras_mnist(categorical=False):
  """
      Loads MNIST Dataset from Keras

      Returns:
          Two tuples representing the training and test datasets
  """
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = np.reshape(x_train, get_input_shape_ones())
  x_test = np.reshape(x_test, get_input_shape_ones())

  if categorical is True:
    y_train = to_categorical(y_train, FLAGS.num_classes)
    y_test = to_categorical(y_test, FLAGS.num_classes)

  return x_train, y_train, x_test, y_test


def sample_keras_mnist(batch_size=FLAGS.batch_size, train_tuple=None):
  """
      Splits the training dataset into batches of batch_size

      Args:
          batch_size: An integer representing the size of the batch
          train_tuple: A tuple containing the training images and training labels
  """

  if train_tuple is None:
    train_tuple = keras_mnist()
    train_tuple = train_tuple[0]

  return sample_data(train_tuple[0], train_tuple[1])


def batch_indices(batch_nb, data_length, batch_size):
  """
      !Adapted from cleverhansl2l Utils!

      This helper function computes a batch start and end index

      Args:
          batch_nb: the batch number
          data_length: the total length of the data being parsed by batches
          batch_size: the number of inputs in each batch

      Returns:
          pair of (start, end) indices
  """
  # Batch start and end index
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)

  # When there are not enough inputs left, we reuse some to complete the
  # batch
  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift

  return start, end


def get_clever_mnist(input_shape=get_input_shape_ones()):
  """Returns the train and
  test data from cleverhansl2l
  :returns: train and test data
  """
  mnist = MNIST(train_start=0, train_end=60000,
                test_start=0, test_end=10000)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')
  x_train = np.reshape(x_train, input_shape)
  x_test = np.reshape(x_test, input_shape)
  return x_train, y_train, x_test, y_test


def get_clever_batch(size=FLAGS.batch_size):
  """
      Args:
          size: size of the batch
      Returns:
          a tuple containing a batch of examples
          and corresponding labels of size size xD
  """
  x_train, y_train, _, _ = get_clever_mnist()
  batch_x = x_train[:size]
  batch_y = y_train[:size]
  return batch_x, batch_y


def get_mnist_unrolls(input_shape=get_input_shape_ones()):
  """Splits and complements data so that it fits
  FLAGS.batch_size * (FLAGS.num_unrolls +1)
  :param input_shape: data is reshaped after this input
  :returns batches_x, batches_y: batches
  """
  rng = np.random.RandomState([2017, 8, 30])
  x_train, y_train, x_test, y_test = get_clever_mnist(input_shape)

  nb_batches = int(math.ceil(float(len(x_train)) / FLAGS.batch_size))
  # Indices to shuffle training set

  index_shuf = list(range(len(x_train)))
  # Randomly repeat a few training examples each epoch to avoid
  # having a too-small batch
  while len(index_shuf) % (FLAGS.batch_size * (FLAGS.truncated_backprop)) != 0:
    index_shuf.append(rng.randint(len(x_train)))
  nb_batches = len(index_shuf) // FLAGS.batch_size
  rng.shuffle(index_shuf)
  # Shuffling here versus inside the loop doesn't seem to affect
  # timing very much, but shuffling here makes the code slightly
  # easier to read
  x_train_shuffled = x_train[index_shuf]
  y_train_shuffled = y_train[index_shuf]

  batches_x = []
  batches_y = []
  for batch in range(nb_batches):
    start = batch * FLAGS.batch_size
    end = (batch + 1) * FLAGS.batch_size
    batches_x.append(x_train_shuffled[start:end])
    batches_y.append(y_train_shuffled[start:end])

  return batches_x, batches_y, x_train, y_train


def get_truncated_data():
  """Prepares training and testing data: every time we do an 'unroll' operation
  we use FLAGS.truncated_backprop data batches. This method splits the
  training and testing data in:
  len(data)/FLGAS/bactch_size/FLAGS.truncated_backprop
  At each iteration we feed the NN FLAGS.truncated_backprop batches
  """
  x_train, y_train, x_test, y_test = get_mnist_unrolls()
  batches_x, batches_y = [], []

  for batch in range(len(x_train) // (FLAGS.truncated_backprop)):
    start = batch * (FLAGS.truncated_backprop)
    end = (batch + 1) * (FLAGS.truncated_backprop)
    batches_x.append(x_train[start:end])
    batches_y.append(y_train[start:end])
  return np.array(batches_x), np.array(batches_y), np.array(x_test), np.array(y_test)
