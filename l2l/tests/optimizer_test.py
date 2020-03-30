"""This module contains the tests for the
optimizer RNN network
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import tensorflow as tf

from l2l.tools.custom_tst import CustomTest
from l2l.optimizer import Optimizer


class TestOptimizer(CustomTest):
  """Unit tests for all Optimizer
  RNN network
  """

  def test_init_optimizer(self):
    optimizer = Optimizer()
    self.assertNotEqual(optimizer, None)
    tf.reset_default_graph()

  def test_build_optimizer(self):
    optimizer = Optimizer()
    cells = optimizer.build()
    self.assertNotEqual(cells, None)

    tf.reset_default_graph()

  def test_get_states(self):
    x_inpt = tf.placeholder(tf.float32, shape=[100, 100])
    optimizer = Optimizer()
    optimizer.build()

    states = optimizer.get_input_states(x_inpt)
    self.assertNotEqual(states, None)

    tf.reset_default_graph()

  def test_run(self):
    x_inpt = tf.placeholder(tf.float32, shape=[None, 784])

    optimizer = Optimizer()
    optimizer.build()

    output, state, deltas = optimizer.run(x_inpt)

    self.assertNotEqual(output, None)
    self.assertNotEqual(state, None)
    self.assertNotEqual(deltas, None)

    tf.reset_default_graph()

  def test_run_multiple(self):
    input_1 = tf.placeholder(tf.float32, shape=[None, 784])
    input_2 = tf.placeholder(tf.float32, shape=[None, 500])

    optimizer = Optimizer()
    optimizer.build()

    output, state, deltas = optimizer.run_multiple_states([input_1, input_2])

    self.assertNotEqual(len(output), 1)
    self.assertNotEqual(len(state), 1)
    self.assertNotEqual(len(deltas), 1)

    tf.reset_default_graph()

  def test_get_variable(self):
    input_1 = tf.placeholder(tf.float32, shape=[None, 784])
    input_2 = tf.placeholder(tf.float32, shape=[None, 500])

    optimizer = Optimizer()
    optimizer.build()

    variables = optimizer.get_variables()

    tf.reset_default_graph()


if __name__ == '__main__':
  unittest.main()
