"""Base class for the meta-optimizer

The optimizer is an LSTM network with 2 hidden
layers
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from l2l import util

FLAGS = tf.flags
FLAGS = FLAGS.FLAGS


class Optimizer():
  """A model for the LSTM optimizer network

  Most parameters for the optimizer are defined
  as flags in flags.py
  """

  def __init__(self):
    """Initializes an optimizer network
    """
    self.cells = None
    self.template = None
    self.default_scope = "multi_rnn_cell/"

  def build(self):
    """Initialises and returns recurrent neural network
    used as optimizer
    :return: returns RNN network built with parameters
      from the flag
    """
    if self.cells is not None:
      return self.cells

    cells = [tf.nn.rnn_cell.LSTMCell(
        num_units=FLAGS.state_size,
        reuse=tf.AUTO_REUSE) for _ in range(FLAGS.rnn_layers)]
    self.cells = tf.nn.rnn_cell.MultiRNNCell(cells)
    return self.cells

  def get_input_states(self, inputs):
    """Returns cell states for several inputs.
    :param inputs: an array of tensors properly shaped
    """
    self.check_cells()
    return self.cells.get_initial_state(inputs)

  def _linear_template(self, name):
    """Creates a reusable template for the last linear layer
    :param name: template name scope
    :return: a Tensorflow template for linear operation
    """
    if self.template:
      return self.template

    def template(inputs, output_size):
      input_shape = inputs.get_shape().as_list()
      w_1 = tf.get_variable(name="W", shape=[input_shape[1], output_size])
      b_1 = tf.get_variable(name="b", shape=[output_size, ])
      return tf.matmul(inputs, w_1) + b_1

    self.template = tf.make_template(
        self.default_scope+name, template, inputs='inputs', output_size='output_size')
    return self.template

  def get_linear(self, inputs):
    """Returns a linear layer
    :param inputs: a tensor with the input data
    :param output_size: an integer representing the size of the hidden layer
    :return: a Tensorflow operation of type Wx + b
    """
    tmplt = self._linear_template('linear')
    return tmplt(inputs=inputs, output_size=1)

  def run(self, inputs, states=None):
    """Runs the optimizer once for given
    inputs and states and returns the results.
    :param inputs: a tensor containing the inputs / gradients
    :param states: an array containing LSTMTuples
    """
    self.check_cells()
    if states is None:
      states = self.get_input_states(inputs)

    with tf.name_scope("preprocess_rnn"):
      reshaped_inputs = util.reshape_inputs(inputs)
      preprocessed = util.preprocess(tf.expand_dims(reshaped_inputs, -1))
      # adds preprocessing data and dimension
      reshaped_inputs = tf.reshape(
          preprocessed, [preprocessed.get_shape().as_list()[0], -1])

    with tf.name_scope("run_rnn"):
      output, new_state = self.cells(reshaped_inputs, states)
      delta = self.get_linear(output)

    with tf.name_scope("postprocess_rnn"):
      delta = delta * FLAGS.optimizer_scale  # add rescaling
      reshaped = util.recover_input_shape(delta, inputs)

    return output, new_state, reshaped

  def run_multiple_states(self, inputs, states=None):
    """Runs the optimizer several times, for each
    input-states pair
    :param inputs: an array of tensors
    :param states: an array of LSTMTuples, or None. If the states
        are none we initialise them here
    :returns: tuple containing the output of the RNN, the new states
        and the output of the fully connected layer
    """
    if states is None:
      states = [self.get_input_states(x) for x in inputs]

    output = np.array([self.run(x, s) for x, s in zip(inputs, states)])
    return output[:, 0], output[:, 1], output[:, 2]

  def update_optimizee(self, loss, variables, states):
    """Updates variables by taking the gradient of the
    loss w.r.t the variables and run it through the RNN,
    then add the output to the variables.
    :param loss: optimizee loss
    :param variables: an array of tensors containing the past
        optimizee variables
    :param states: LSTM states
    :return: tuple containing new variables and list of new
        LSTM states
    """
    with tf.name_scope("gradients"):
      grads = util.get_grads(loss, variables)

    with tf.name_scope("deltas"):
      _, new_states, deltas = self.run_multiple_states(grads, states)

    with tf.name_scope("add_deltas"):
      new_variables = [x + delta for x, delta in zip(variables, deltas)]

    return new_variables, list(new_states)

  def optimize(self, loss):
    """Optimizes a given loss only w.r.t the optimizer variables
    :param loss: loss function
    :returns: minimize operation
    """
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    optimizer_vars = self.get_variables()
    if FLAGS.instrumentation:
      print('[INFO] Feeding variables to the optimizer: {}'.format(optimizer_vars))
    return optimizer.minimize(loss, var_list=optimizer_vars)

  def check_cells(self):
    """Checks if the optimizer has
    been built before
    """
    if self.cells is None:
      self.build()

  def get_variables(self):
    """Returns optimizer variables
    """
    optimizer_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES,
        scope=self.default_scope
    )
    return optimizer_vars
