from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

flags = tf.flags
FLAGS = flags.FLAGS


class Optimizee(object):
  def __init__(self, **kwargs):
    if kwargs is not None:
      self.config = kwargs

  def evaluate(self, x):
    scale = tf.random_uniform([FLAGS.problem_dims], 0.5, 0.5)
    x = scale*x
    return tf.reduce_sum(x*x)
