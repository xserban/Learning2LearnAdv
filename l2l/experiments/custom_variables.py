from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mock
import tensorflow as tf

from l2l import util

flags = tf.flags
FLAGS = flags.FLAGS


def get_var():
  return tf.get_variable('0', shape=[-1, 784], dtype=tf.float32)


def get_var_custom():
  with mock.patch('tensorflow.get_variable', util.mock_get_var):
    v = get_var()
    return v


def inference(input):
  v = get_var()


def inference_custom():
  x = tf.placeholder(dtype=tf.float32, shape=util.get_input_shape_none())
  variables = [tf.Variable(initial_value=0)]

  inf_op_custom = util.make_with_custom_variables(inference, [x], variables)

  return inf_op_custom
