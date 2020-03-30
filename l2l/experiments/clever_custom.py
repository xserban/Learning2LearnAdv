from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cleverhansl2l.model import Model
from cleverhansl2l.picklable_model import MLP, Linear, ReLU
from cleverhansl2l.picklable_model import Softmax
from cleverhansl2l.dataset import MNIST
from l2l import util

flags = tf.flags
FLAGS = flags.FLAGS


input_shape = (None, 784)
layers = [Linear(FLAGS.num_units, scope="linear_1"),
          ReLU(),
          Linear(FLAGS.num_units, scope="linear_2"),
          ReLU(),
          Linear(FLAGS.num_classes, scope="linear_3"),
          Softmax()
          ]


def create_clever_model():
  model = MLP(layers, input_shape)
  return model


def get_variables_clever_custom():
  op = MLP
  kwargs = {'layers': layers, 'input_shape': input_shape}
  variables, _ = util.get_variables(op, kwargs)
  return variables


def clever_custom_vars():
  op = MLP
  kwargs = {'layers': layers, 'input_shape': input_shape}
  variables, _ = util.get_variables(op, kwargs)

  custom_model = util.make_with_custom_variables(op, kwargs, variables)
  return custom_model


def run_clever_model():
  model = MLP(layers, input_shape)

  mnist = MNIST(train_start=0, train_end=60000,
                test_start=0, test_end=10000)
  x_train, y_train = mnist.get_set('train')
  x_test, y_test = mnist.get_set('test')

  # do some simple reshaping
  x_train = np.reshape(x_train, [-1, 784])
  x_test = np.reshape(x_test, [-1, 784])

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, 784))
  y = tf.placeholder(tf.float32, shape=(None, 10))

  logits = model.get_logits(x)
  losses = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=y, logits=logits)
  loss = tf.reduce_mean(losses)

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)
    feed_dict = {x: x_train[:128], y: y_train[:128]}
    loss = sess.run(loss, feed_dict=feed_dict)

    return loss
