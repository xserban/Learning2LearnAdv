from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

import tensorflow as tf

from l2l.tools.custom_tst import *
from l2l.experiments.clever_custom import *


class TestCleverhansExp(CustomTest):
  def setUp(self):
    return super().setUp()

  def test_create_clever_model(self):
    model = create_clever_model()
    self.assertNotEqual(model, None)
    tf.reset_default_graph()

  def test_get_variables_clever_custom(self):
    variables = get_variables_clever_custom()
    self.assertGreater(len(variables), 0)
    tf.reset_default_graph()

  def test_custom_var_clever(self):
    model = clever_custom_vars()
    self.assertTrue(True)
    tf.reset_default_graph()

  def test_run_clever_model(self):
    loss = run_clever_model()
    self.assertGreater(loss, 0)

  def tearDown(self):
    return super().tearDown()


if __name__ == '__main__':
  unittest.main()
