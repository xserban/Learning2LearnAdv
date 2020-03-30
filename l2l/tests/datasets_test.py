from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

import tensorflow as tf

from l2l.tools.custom_tst import *
from l2l.util import *
from l2l.datasets import *


class TestDatasetsMethods(CustomTest):
  def setUp(self):
    return super().setUp()

  def test_get_mnist(self):
    _data, _images, _labels = get_mnist()
    self.assertEqual(_images.shape[0], _labels.shape[0])

  def test_sample_mnist(self):
    batch_samples, batch_labels = sample_mnist()
    # check if the number of samples & labels is equal
    self.assertEqual(batch_samples.shape[0], batch_labels.shape[0])

  def test_keras_mnist(self):
    train_tuple, test_tuple = keras_mnist()
    # check if the number of samples & labels is equal
    self.assertEqual(train_tuple[0].shape[0], train_tuple[0].shape[0])
    self.assertEqual(train_tuple[1].shape[0], train_tuple[1].shape[0])

  def test_batch_keras_mnist(self):
    batch_samples, batch_labels = sample_keras_mnist()
    self.assertEqual(batch_samples.shape[0], batch_labels.shape[0])

  def test_clever_batch(self):
    batch_x, batch_y = get_clever_batch()
    self.assertEqual(batch_x.shape[0], batch_y.shape[0])

  def test_get_mnist_unroll(self):
    x_train, y_train, _, _ = get_mnist_unrolls()
    self.assertTrue(len(x_train), len(y_train))
    self.assertGreater(len(x_train), 0)

  def test_get_truncated_data(self):
    batches_x, batches_y, _, _ = get_truncated_data()
    self.assertAlmostEqual(batches_x.shape[0], batches_y.shape[0])

  def tearDown(self):
    return super().tearDown()


if __name__ == '__main__':
  unittest.main()
