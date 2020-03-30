from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

import numpy as np
import tensorflow as tf

from l2l.tools.custom_tst import *
from l2l.experiments.clever_recursive import *


class TestCleverRecursive(CustomTest):
  def setUp(self):
    return super().setUp()

  def test_create_tf_loop(self):
    fx_array = create_tf_loop()
    self.assertTrue(True)

  def test_run_tf_loop(self):
    loss = run_tf_loop()
    self.assertGreater(loss, 0)

  def tearDown(self):
    return super().tearDown()


if __name__ == '__main__':
  unittest.main()
