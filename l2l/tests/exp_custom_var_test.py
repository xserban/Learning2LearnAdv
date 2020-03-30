from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

import tensorflow as tf

from l2l.tools.custom_tst import *
from l2l.experiments.custom_variables import *


class TestDatasetsMethods(CustomTest):
  def setUp(self):
    return super().setUp()

  def test_custom_var(self):
    var = get_var_custom()
    self.assertEqual(var, None)

  def test_custom_inference(self):
    op = inference_custom()
    self.assertTrue(True)

  def tearDown(self):
    return super().tearDown()


if __name__ == '__main__':
  unittest.main()
