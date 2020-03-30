
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import mock
import json
import numpy as np
import tensorflow as tf


from tensorflow.contrib.learn.python.learn.datasets import mnist as mnist_dataset
from l2l.constants import AdversarialConstants

# import flags in util
# so they can be used everywhere util is imported
from l2l import flags

flags = tf.flags
FLAGS = flags.FLAGS

ADV_CONSTANTS = AdversarialConstants()


def get_weight(shape):
  return np.random.rand(shape)


def get_bias(shape):
  return np.zeros(shape)


def get_input_shape_none():
  """ Returns:
          input shape from flags
  """
  if FLAGS.input_shape:
    return [int(dim) if dim != '-1' else None for dim in FLAGS.input_shape.split(',')]


def get_input_shape_ones():
  """ Returns:
          input shape from flags
  """
  if FLAGS.input_shape:
    return [int(dim) if dim != '-1' else -1 for dim in FLAGS.input_shape.split(',')]


def replace_none_shape(shape):
  """Replaces None with -1 in shape
  :param shape: array with initial shape components
  :returns: array with new shape
  """
  return [-1 if x is None else x for x in shape]


def get_vars(func, args, scope='useless_graph'):
  """Calls func, returning any variables created, but ignoring its return value.

  Args:
    func: Function to be called.

  Returns:
    A tuple (variables, constants) where the first element is a list of
    trainable variables and the second is the non-trainable variables.
  """
  variables = []
  constants = []

  def custom_getter(getter, name, **kwargs):
    trainable = kwargs["trainable"]
    kwargs["trainable"] = False
    variable = getter(name, **kwargs)
    if trainable:
      variables.append(variable)
    else:
      constants.append(variable)
    return variable

  with tf.name_scope(scope):
    wrap_variable_creation(func, args, custom_getter)

  return variables, constants


def vars_to_dic(vars):
  """ Converts list of tf variables to a dictionary
      where the var names are the keys

  Args:
      vars: list of variables

  Returns:
      dic
  """
  return dict((var.name.split(":")[0], i)
              for i, var in enumerate(vars))


def create_hidden_states(coordinates):
  """ Creates a RNN hidden state for each coordinate
      given.

      Args:
          coordinates: array of variables corresponding to
              each optimizee parameter.

      Returns:
          An array of variables indexed in the same order as
          the coordinates
  """
  hidden_states = []

  with tf.name_scope("hidden_states"):
    for inpt in coordinates:
      states = []

      var_name = '/'.join(inpt.name.split('/')[-2:]).replace(':', '/')
      reshaped_inpt = tf.reshape(inpt, [-1, 1])
      batch_size = reshaped_inpt.get_shape().as_list()[0]

      with tf.name_scope("{}/cell_1".format(var_name)):
        h = tf.Variable(
            np.zeros([batch_size, FLAGS.state_size]), tf.float32)
        states.append(tf.nn.rnn_cell(LSTMStateTuple(c=h, h=h))
                      )  # pylint: disable=not-callable

      with tf.name_scope("{}/cell_2".format(var_name)):
        h = tf.Variable(
            np.zeros([batch_size, FLAGS.state_size]), tf.float32)
        states.append(tf.nn.rnn_cell(LSTMStateTuple(c=h, h=h))
                      )  # pylint: disable=not-callable

      hidden_states.append(states)

  return hidden_states

# The following 2 methods are adapted from
# Google DeepMind L2L Code


def make_with_custom_variables(fx, fkargs, variables):
  """
      Creates operations & tensors in the graph with the
      given variables
  """
  variables = collections.deque(variables)

  def custom_getter(getter, name, **kwargs):
    if kwargs["trainable"]:
      return variables.popleft()
    else:
      kwargs["reuse"] = True
      return getter(name, **kwargs)

  return wrap_variable_creation(fx, fkargs, custom_getter)


def wrap_variable_creation(fx, fkargs, custom_getter):
  """
      Provides a custom getter for all variables
  """
  original_get_variable = tf.get_variable

  def custom_get_variable(*args, **kwargs):
    if hasattr(kwargs, "custom_getter"):
      raise AttributeError("Custom getters are not supported for optimizee "
                           "variables.")
    return original_get_variable(*args, custom_getter=custom_getter, **kwargs)

  # Mock the get_variable method.
  with mock.patch("tensorflow.get_variable", custom_get_variable):
    return fx(**fkargs)

# End of methods from DeepMind


def mock_get_var(*args, **kwargs):
  print('Mocking Tensorflow Get Variable')
  return None


def get_grads(function, variables):
  """
      Returns the gradients of the function w.r.t
      the variables
      Args:
          function: a Tensorflow operation
          variables: an arrayof variables for taking the
          gradients
  """
  grads = tf.gradients(function, variables)
  if not FLAGS.second_derivatives:
    grads = [tf.stop_gradient(g) for g in grads]
  return grads


def reshape_inputs(inputs):
  """ Reshapes an input for the RNN to batch
      size -1 and one element / time-step
  """
  return tf.reshape(inputs, [-1, 1])


def recover_input_shape(outputs, inputs):
  """
      Reshape output to input shape
  """
  input_shape = [x if x is not None else
                 -1 for x in inputs.get_shape().as_list()]
  return tf.reshape(outputs, input_shape)


def get_lstm_state_tuples(stacked_cells, input):
  """
      Returns list of lstm states
  """
  return stacked_cells.get_initial_state(inputs=reshape_inputs(input))


def reinit_vars(var_list):
  """
      Reinitialize variables from list
      This method is currently a wrapper
      around tf.initializers

      However, this wrapper was created
      in case we need some preprocessing
  """
  return tf.initializers.variables(var_list)

# The following methods are taken from Deep Mind
# Learning to learn


def nested_assign(ref, value):
  """Returns a nested collection of TensorFlow assign operations.

  Args:
    ref: Nested collection of TensorFlow variables.
    value: Values to be assigned to the variables. Must have the same structure
        as `ref`.

  Returns:
    Nested collection (same structure as `ref`) of TensorFlow assign operations.

  Raises:
    ValueError: If `ref` and `values` have different structures.
  """
  if isinstance(ref, list) or isinstance(ref, tuple):
    if len(ref) != len(value):
      raise ValueError("ref and value have different lengths.")
    result = [nested_assign(r, v) for r, v in zip(ref, value)]
    if isinstance(ref, tuple):
      return tuple(result)
    return result
  else:
    return tf.assign(ref, value)


def nested_variable(init, name=None, trainable=False):
  """Returns a nested collection of TensorFlow variables.

  Args:
    init: Nested collection of TensorFlow initializers.
    name: Variable name.
    trainable: Make variables trainable (`False` by default).

  Returns:
    Nested collection (same structure as `init`) of TensorFlow variables.
  """
  if isinstance(init, list) or isinstance(init, tuple):
    result = [nested_variable(i, name, trainable) for i in init]
    if isinstance(init, tuple):
      return tuple(result)
    return result
  else:
    return tf.Variable(init, name=name, trainable=trainable)

# Preprocessing steps are taken from
# DeepMind L2L repo


def clamp(inputs, min_value=None, max_value=None):
  """
  :param inputs: a tensor which contains the gradients
  :param min_val: integer for lower bound
  :param max_val: integer for upper bound
  :returns: a tensor clipped between the lower and upper bound
  """
  if min_value is not None:
    outputs = tf.maximum(inputs, min_value)
  if max_value is not None:
    outputs = tf.minimum(outputs, max_value)
  return outputs


def log_sign(gradients, k=FLAGS.k):
  """
  :param gradients: gradients of the optimizee
  :param k:
  :returns: a tensor with preprocessed gradients with log sign
  """
  eps = np.finfo(gradients.dtype.as_numpy_dtype).eps
  ndims = gradients.get_shape().ndims

  log = tf.log(tf.abs(gradients) + eps)
  preprocess_grads = log/k
  clamped_log = clamp(preprocess_grads, min_value=-1.0)
  preprocess_sign = gradients*np.exp(k)
  sign = clamp(preprocess_sign, min_value=-1.0,
               max_value=1.0)

  return tf.concat([clamped_log, sign], ndims-1)


def multiply(gradients, k=FLAGS.k):
  """ Preprocess inputs by multiplying with a constant
  :param gradients: a tensor containing the gradients
  :param k: an integer representing the preprocessing constant/step
  :returns: a tensor with preprocessed gradients
  """
  ndims = gradients.get_shape().ndims
  mlt = gradients * k
  sign = clamp(min_value=-1.0, max_value=1.0, inputs=gradients*np.exp(k))
  return tf.concat([mlt, sign], ndims-1)


def preprocess(inputs):
  """Preprocesses inputs with preprocessing
  :param inputs: input tensor
  :returns: tensorflow operation for reshaping
  """
  with tf.name_scope("preprocess"):
    # simulating aswitch between possible preprocessing methods
    if FLAGS.preprocessing is 'log_sign':
      return log_sign(tf.expand_dims(inputs, -1))
    elif FLAGS.preprocessing is 'multiply':
      return multiply(tf.expand_dims(inputs, -1))


def save_grad_histo(gradients):
  """Saves a summary histogram for each gradient
  :param gradients: list of tensors with gradient values
  """
  if FLAGS.instrumentation and FLAGS.print_grads:
    for i, g in enumerate(gradients):
      tf.summary.histogram("gradients/" + str(i), g)


def save_grad_mean(gradients):
  """Saves a summary with the mean for each grad
  :param gradients: list of tensors with gradient values
  """
  if FLAGS.instrumentation and FLAGS.print_grads:
    for i, g in enumerate(gradients):
      tf.summary.scalar('gradients/' + str(i), tf.reduce_mean(g))


def get_log_dir(train=True):
  """Returns formatted directory for tensorboard"""
  adv = 'adversarial/' if FLAGS.adversarial is True else ''
  attack = adv + FLAGS.train_attack if train is True else FLAGS.test_attack
  clipped = attack + ('clipped/' if FLAGS.clipped else '')
  eps = clipped + "/" + (ADV_CONSTANTS.get_attack_detail(
      FLAGS.train_attack, "eps") if FLAGS.adversarial is True else '')

  final_dir = FLAGS.preprocessing + "/" + str(FLAGS.k) + '/' + eps
  return final_dir


def is_same(t_1, t_2, norm=np.inf):
  """Compares two vectors and returns a boolean
  if they are equal and their difference in norm
  :param t_1: first tensor
  :param t_2: second tensor
  :param norm: size of the difference
  """
  is_same = tf.equal(t_1, t_2)
  diff = t_2 - t_1
  return is_same, tf.norm(diff, ord=norm)
