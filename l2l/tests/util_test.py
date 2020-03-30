from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

import tensorflow as tf

from l2l.tools.custom_tst import *
from l2l.util import *
from l2l.datasets import *
from l2l.optimizee_fc import OptimizeeFC


class TestUtilMethods(CustomTest):
  def setUp(self):
    super().setUp()
    self.optimizee = OptimizeeFC()

  def test_get_input_shape_none(self):
    input_shape = get_input_shape_none()
    self.assertTrue(len(input_shape) > 0)

  def test_get_input_shape_one(self):
    input_shape = get_input_shape_none()
    self.assertTrue(len(input_shape) > 0)

  def test_replace_none_shape(self):
    test_shape = [None, 784]
    reshaped = replace_none_shape(test_shape)
    test = None
    for _, v in enumerate(reshaped):
      if v is None:
        test = True
        return
    self.assertEqual(test, None)

  def test_get_variables(self):
    inpt = tf.placeholder(dtype=tf.float32, shape=get_input_shape_none())
    variables, _ = get_vars(self.optimizee.inference, {
        'input': inpt, 'save_logits': False})
    # easy assertion to see if the graph is formed
    self.assertTrue(len(variables) > 0)
    tf.reset_default_graph()

  def test_vars_to_dic(self):
    inpt = tf.placeholder(dtype=tf.float32, shape=get_input_shape_none())
    variables, _ = get_vars(self.optimizee.inference, {
        'input': inpt, 'save_logits': False})

    dic = vars_to_dic(variables)
    self.assertTrue(isinstance(dic, dict))
    tf.reset_default_graph()

  def test_create_hidden_states(self):
    inpt = tf.placeholder(dtype=tf.float32, shape=get_input_shape_none())
    variables, _ = get_vars(self.optimizee.inference, {
        'input': inpt, 'save_logits': False})

    hidden_states = create_hidden_states(variables)
    self.assertEqual(len(variables), len(hidden_states))
    tf.reset_default_graph()

  def test_get_grads(self):
    pass

  def test_reshape_inputs(self):
    pass

  def get_lstm_state_tuples(self):
    pass

  def tearDown(self):
    super().tearDown()
    tf.reset_default_graph()


if __name__ == '__main__':
  unittest.main()
