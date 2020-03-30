from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


class Optimizer(object):
  def __init__(self, **kwargs):
    """ Initializes optimizer object - in this
        case a RNN

        Config is empty at the moment because
        we use tensorflow flags
    """
    if kwargs is not None:
      self.config = kwargs
    else:
      raise ValueError(
          'Configuration parameters needed to initialise Optimizer.')

    self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

  def inference(self):
    """ Returns RNN """
    # configure number of layers
    cells = [tf.nn.rnn_cell.LSTMCell(
        name='basic_lstm_cell',
        num_units=FLAGS.state_size,
        state_is_tuple=FLAGS.state_tuple)
        for _ in range(FLAGS.rnn_layers)]

    # create rnn
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    # create input & output projection layers
    input_proj = tf.contrib.rnn.InputProjectionWrapper(
        stacked_cells, FLAGS.state_size)

    output_proj = tf.contrib.rnn.OutputProjectionWrapper(input_proj, 1)

    self.rnn = tf.make_template('cell', output_proj)
    return self.rnn

  def loss(self):
    pass

  def optimize(self, loss):
    """ Applies gradients """
    gradients, v = zip(*self.optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.)
    return self.optimizer.apply_gradients(zip(gradients, v))

  def evaluate(self, gradients, state):
    """ Returns the update for optimizee """

    gradients = tf.expand_dims(gradients, axis=1)
    if state is None:
      state = [
          [tf.zeros([FLAGS.problem_dims, FLAGS.state_size])]*2] * FLAGS.rnn_layers

    update, state = self.rnn(gradients, state)
    return tf.squeeze(update, axis=[1]), state
