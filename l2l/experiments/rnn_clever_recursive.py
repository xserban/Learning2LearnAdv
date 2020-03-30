from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn import monitored_session as ms
from tensorflow.python.util import nest

from l2l import util
from l2l import datasets

from cleverhansl2l.model import Model
from cleverhansl2l.picklable_model import MLP, Linear, ReLU
from cleverhansl2l.picklable_model import Softmax

flags = tf.flags
FLAGS = flags.FLAGS

# Cleverhans Model Config
layers = [Linear(FLAGS.num_units, scope="linear"),
          ReLU(),
          Linear(FLAGS.num_classes, scope="logits"),
          ]
input_shape = (None, 784)

op = MLP
opkwargs = {'layers': layers, 'input_shape': input_shape}


def get_model_loss(x, y, custom_variables=None):
  if custom_variables is None:
    model = op(**opkwargs)
    logits = model.get_logits(x)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=logits)
    return loss

  model = util.make_with_custom_variables(op, opkwargs, custom_variables)
  logits = model.get_logits(x)
  loss = tf.nn.softmax_cross_entropy_with_logits_v2(
      labels=y, logits=logits)
  return loss


def get_optimizer():
  with tf.variable_scope("rnn"):
    cells = [tf.nn.rnn_cell.LSTMCell(
        num_units=FLAGS.state_size, reuse=tf.AUTO_REUSE) for _ in range(FLAGS.rnn_layers)]
    stacked_cells = tf.nn.rnn_cell.MultiRNNCell(cells)
  # instead of linear we can use the outputprojectionwrapper
  return stacked_cells


def get_input_states(optimizer, inputs):
  reshaped_inputs = [util.reshape_inputs(x) for x in inputs]
  return [util.get_lstm_state_tuples(optimizer, x) for x in reshaped_inputs]


def get_linear(input, output_size):
  input_shape = input.get_shape().as_list()
  with tf.variable_scope("final_linear", reuse=tf.AUTO_REUSE):
    W = tf.get_variable(name="final_W", shape=[
                        input_shape[1], output_size])
    b = tf.get_variable(name="final_b", shape=[output_size])
    mul = tf.matmul(input, W) + b

  return mul


def run_optimizer(inputs, states=None):
  optimizer = get_optimizer()
  reshaped_inputs = util.reshape_inputs(inputs)

  if states is None:
    states = util.get_lstm_state_tuples(optimizer, reshaped_inputs)

  output, new_state = optimizer(reshaped_inputs, states)

  delta = get_linear(output, 1)
  return output, new_state, util.recover_input_shape(delta, inputs)


def run_optimizer_for_different_states(inputs, states=None):
  if states is None:
    states = [None for _ in range(len(inputs))]

  output = [run_optimizer(
      x, s) for x, s in zip(inputs, states)]
  output = np.array(output)

  # returns in this order: output of RNN, new states for RNN
  # and deltas
  return output[:, 0], output[:, 1], output[:, 2]


def get_updated_variables(loss, current_variables, current_states):
  with tf.name_scope("gradients"):
    grads = util.get_grads(loss, current_variables)

  with tf.name_scope("deltas"):
    outputs, next_states, deltas = run_optimizer_for_different_states(
        grads, current_states)

  next_variables = [x + delta for x, delta in zip(current_variables, deltas)]

  return next_variables, list(next_states)


def loop_body(t, input_arguments, loss_array, current_variables, current_states=None):
  # creates a new loss function with the variables
  # updated at the last iteration of the loop and
  # then updates the variables by running the optimizer
  # on the gradients of the loss taken w.r.t variables
  if current_states == []:
    current_states = None

  with tf.name_scope("model_loss_new_vars"):
    current_loss = get_model_loss(
        custom_variables=current_variables, **input_arguments)
    loss_array = loss_array.write(t, current_loss)

  with tf.name_scope("update_loss"):
    next_variables, next_states = get_updated_variables(
        current_loss, current_variables, current_states)

  with tf.name_scope("time_step"):
    time_step = t + 1

  return time_step, input_arguments, loss_array, next_variables, next_states


def run_loop(input_arguments, current_variables, current_constants):
  optimizer = get_optimizer()

  input_states = get_input_states(optimizer, current_variables)

  loss_array = tf.TensorArray(
      tf.float32,
      size=FLAGS.truncated_backprop+1,
      clear_after_read=False)

  _, _, loss_array, final_variables, final_states = tf.while_loop(
      cond=lambda t, *_: t < FLAGS.truncated_backprop,
      body=loop_body,
      loop_vars=(0, input_arguments, loss_array,
                 current_variables, input_states),
      name="unroll"
  )

  # run last time without updating
  # the variables
  with tf.name_scope("final_loss"):
    final_loss = get_model_loss(
        custom_variables=final_variables, **input_arguments)
    loss_array = loss_array.write(FLAGS.truncated_backprop, final_loss)

  loss = tf.reduce_sum(loss_array.stack(), name="optimizee_loss")
  tf.summary.scalar("cost_function", loss)

  # reset all variables for passing
  # the states around
  with tf.name_scope("reset_loop_states"):
    variables = (nest.flatten(util.nested_variable(input_states)))
    # Empty array as part of the reset process.
    reset_op = [tf.variables_initializer(variables), loss_array.close()]

  # update the final vars
  with tf.name_scope("update"):
    update_op = (nest.flatten(util.nested_assign(current_variables, final_variables)) +
                 nest.flatten(util.nested_assign(util.nested_variable(input_states), final_states)))

  return loss, reset_op, update_op, final_variables, final_loss


def optimize(loss):
  optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
  step = optimizer.minimize(loss)
  return step


def run_epoch(sess,  reset, cost, update, optimise, dic):
  sess.run(reset)
  sess.run([cost, update, optimise], feed_dict=dic)


def reset_rnn_states():
  pass
