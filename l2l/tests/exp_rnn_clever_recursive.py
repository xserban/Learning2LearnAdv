from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import os

import numpy as np
import tensorflow as tf

from l2l import util
from l2l.tools.custom_tst import *
from l2l.datasets import *
from l2l.experiments.rnn_clever_recursive import *


class TestRnnCleverRecursive(CustomTest):
  def setUp(self):
    return super().setUp()

  def test_call_optimizer_twice(self):
    cells = get_optimizer()
    cells_2 = get_optimizer()

    self.assertTrue(True)
    tf.reset_default_graph()

  def test_run_optimizer_build(self):
    # we only build the graph, but never run it
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 10])

    coordinates, _ = util.get_variables(get_model_loss, {'x': x,
                                                         'y': y,
                                                         'custom_variables': None})
    loss = get_model_loss(x, y, custom_variables=coordinates)
    grads = util.get_grads(loss, coordinates)

    optimizer = get_optimizer()
    hidden = util.get_lstm_state_tuples(optimizer, grads[0])

    output, state = optimizer(util.reshape_inputs(grads[0]), hidden)
    self.assertTrue(True)

    tf.reset_default_graph()

  def test_run_optimizer(self):
    # we just build the graph here
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 10])

    coordinates, _ = util.get_variables(get_model_loss, {'x': x,
                                                         'y': y,
                                                         'custom_variables': None})
    loss = get_model_loss(x, y, custom_variables=coordinates)
    grads = util.get_grads(loss, coordinates)

    output, state, delta = run_optimizer(grads[0])
    self.assertTrue(True)

    tf.reset_default_graph()

  def test_run_optimizer_different_states(self):
    # test if the graph can be built without any
    # errors
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 10])

    coordinates, _ = util.get_variables(get_model_loss, {'x': x,
                                                         'y': y,
                                                         'custom_variables': None})
    loss = get_model_loss(x, y, custom_variables=coordinates)
    grads = util.get_grads(loss, coordinates)

    outputs, states, delta = run_optimizer_for_different_states(grads)

    self.assertTrue(True)
    tf.reset_default_graph()

  def test_run_loop(self):
    # build the complete running loop graph
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
    input_arguments = {'x': x, 'y': y}

    coordinates, _ = util.get_variables(get_model_loss, {'x': x,
                                                         'y': y,
                                                         'custom_variables': None})
    model_loss = get_model_loss(x, y, custom_variables=coordinates)

    loop_loss = run_loop(input_arguments, coordinates)
    self.assertTrue(True)

  def test_run_one_iteration_loop(self):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 10])
    input_arguments = {'x': x, 'y': y}

    coordinates, constants = util.get_variables(get_model_loss, {'x': x,
                                                                 'y': y,
                                                                 'custom_variables': None})

    loss, reset_loop_vars, update_op, final_vars, final_loss = run_loop(
        input_arguments, coordinates, constants)

    with tf.name_scope("reset"):
      vars = coordinates
      if len(constants) > 0:
        vars = np.concatenate(coordinates, constants)
      reset_op = util.reinit_vars(vars)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      # write graph to disk
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter('./logs', sess.graph)
      sess.run(init)
      tf.get_default_graph().finalize()

      # run the loop loss only once
      # with some random batch of examples
      batch_x, batch_y, = get_clever_batch()
      sess.run([loss, update_op], feed_dict={x: batch_x, y: batch_y})

      self.assertTrue(True)

  def test_run_several_epochs(self):
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
    y = tf.placeholder(dtype=tf.int32, shape=[None, 10])

    input_arguments = {'x': x, 'y': y}

    variables, constants = util.get_variables(get_model_loss, {'x': x,
                                                               'y': y,
                                                               'custom_variables': None})
    all_losses, reset_loop_vars, update_op, final_vars, final_loss = run_loop(
        input_arguments, variables, constants)
    optimize_op = optimize(all_losses)

    with tf.name_scope("reset"):
      vars = constants
      if len(constants) > 0:
        vars = np.concatenate(variables, constants)
      reset_var_op = util.reinit_vars(vars)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      for _ in range(FLAGS.num_epochs_test):
        batch_x, batch_y, = get_clever_batch()
        dic = {x: batch_x, y: batch_y}
        run_epoch(sess, [reset_loop_vars, reset_var_op], final_loss,
                  update_op, optimize_op, dic)

  def tearDown(self):
    return super().tearDown()


if __name__ == '__main__':
  unittest.main()
