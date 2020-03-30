"""This module contains all unit tests
for the optimizer fully connected network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import tensorflow as tf

from l2l.tools.custom_tst import CustomTest
from l2l.optimizee_fc import OptimizeeFC


class TestOptimizeeFC(CustomTest):
  """Unit tests for the Optimizee
  fully connected network
  """

  def test_init_optimizee(self):
    opt = OptimizeeFC()
    self.assertNotEqual(opt, None)
    tf.reset_default_graph()

  def test_build_optimizee(self):
    x_inpt = tf.placeholder(tf.float32, shape=[None, 784])
    labels = tf.placeholder(tf.int16, shape=[None, 10])

    opt = OptimizeeFC()
    loss = opt.loss(x_inpt, labels)
    self.assertNotEqual(loss, None)
    tf.reset_default_graph()


if __name__ == '__main__':
  unittest.main()
