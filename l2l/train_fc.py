"""This module contains all methods needed to run
learning-to-learn by gradient descentby gradient descent
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.python.util import nest

from cleverhansl2l.utils import AccuracyReport

from l2l import util
from l2l import datasets
from l2l.adversarial.models.custom_model import CustomModel
from l2l.constants import OptimizeeConstants, AdversarialConstants
from l2l.logger import Logger
from l2l.optimizee_fc import OptimizeeFC
from l2l.optimizer import Optimizer
import l2l.adversarial.util as adv_util

FLAGS = tf.flags
FLAGS = FLAGS.FLAGS

MODEL_CONSTANTS = OptimizeeConstants()
ADV_CONSTANTS = AdversarialConstants()


def random_batch():
  """Returns a random batch of data+labels
  It is mainly used within the "final_loss" scope in
  unroll
  :returns: batch_images, batch_labels
  """
  x_train, y_train, _, _ = datasets.get_clever_mnist(
      input_shape=[-1, 784])

  with tf.variable_scope("dataset", reuse=tf.AUTO_REUSE):
    images = tf.constant(x_train, dtype=tf.float32, name="MNIST_images")
    labels = tf.constant(y_train, dtype=tf.int64, name="MNIST_labels")
  indices = tf.random_uniform(
      [FLAGS.batch_size], 0, len(x_train), tf.int64, seed=1)

  batch_images = tf.gather(images, indices)
  batch_labels = tf.gather(labels, indices)

  return batch_images, batch_labels


def loop_body(time, input_args, loss_array,
              variables, states=None):
  """Creates a new loss with the variables updated in the last
  iteration of the loop and updates the variables by running the
  optimizer on the gradietns taken w.r.t the variables
  :param time: integer which keeps the loop iteration
  :param input_args: input arguments for new loss function
  :param loss_array: a tensorflow array where we store the outcome of the
      loss at each iteration
  :param variables: variables from the last step of the iteration
  :param states: LSTM states from past iteration
  :returns: a tuple with the same parameters as the input arguments
  """
  optimizee = OptimizeeFC()
  optimizer = Optimizer()

  # copy variables because generating an adversarial
  # example will stop the gradients flowing to the inputs
  with tf.variable_scope("copy_input", reuse=tf.AUTO_REUSE):
    cp_inputs = tf.identity(input_args['inputs'][time], name="input_copy")
    cp_labels = tf.identity(input_args['labels'][time], name='labels_copy')

  with tf.variable_scope("adversarial"):
    adv_inputs = optimizee.generate_adv(
        input_args['inputs'][time])  # nput_args['labels']

  args = {'inputs': adv_inputs if FLAGS.adversarial else cp_inputs,
          'labels': cp_labels}

  with tf.name_scope("new_model_loss"):
    new_loss = optimizee.loss(**args, custom_variables=variables)
    loss_array = loss_array.write(time, new_loss)

  with tf.name_scope("update_loss"):
    new_variables, new_states = optimizer.update_optimizee(
        new_loss, variables, states)

  with tf.name_scope("time_step"):
    time_step = time + 1

  return time_step, input_args, loss_array, new_variables, new_states


def unroll(input_arguments):
  """Creates the unroll operations and returns the final loss,
  variables and aditional update and reset operations used
  to start a new training epoch.
  :param input_arguments: dictionary containing arguments for
      optimizee loss
  :returns: a tuple containing the final loss (reduce sum),
      reset and update operations, the final coordinates
      (or variables) and the final loss operation
  """
  optimizer = Optimizer()

  # get list of variables and constants
  fake_optee = OptimizeeFC()
  variables, constants = util.get_vars(fake_optee.loss,
                                       {'inputs': tf.placeholder(tf.float32,
                                                                 shape=[None, 784]),
                                        'labels': tf.placeholder(tf.int32,
                                                                 shape=[None, 10])},
                                       scope=MODEL_CONSTANTS.FINAL_SCOPE)
  # get and reshape RNN states for each variable
  with tf.name_scope("states"):
    reshaped_variables = [util.reshape_inputs(x) for x in variables]
    initial_states = [optimizer.get_input_states(
        x) for x in reshaped_variables]
  # initialize array for keeping track of the loss values in loomp
  loss_array = tf.TensorArray(tf.float32, size=FLAGS.truncated_backprop+1,
                              clear_after_read=False)
  # run loop and collect the loss values in the array
  _, _, loss_array, new_variables, new_states = tf.while_loop(
      cond=lambda t, *_: t < FLAGS.truncated_backprop,
      body=loop_body,
      loop_vars=(0, input_arguments, loss_array, variables, initial_states),
      swap_memory=True,
      parallel_iterations=1,
      name="unroll")

  # run last time without any gradient update
  with tf.name_scope("final_loss"):
    fl_optimizee = OptimizeeFC()
    random_x, random_y = random_batch()
    args = {'inputs': random_x, 'labels': random_y}
    final_loss = fl_optimizee.loss(
        custom_variables=new_variables, **args)
    loss_array = loss_array.write(FLAGS.truncated_backprop, final_loss)
  # print some logs: the gradient histogram so we see how they
  # evolved and the difference in vectors if they are adversarial
  if FLAGS.instrumentation:
    grads = util.get_grads(final_loss, new_variables)
    util.save_grad_mean(grads)
    if FLAGS.adversarial:
      adv_ex = fl_optimizee.generate_adv(random_x)
      _, diff = util.is_same(adv_ex, random_x)
      tf.summary.scalar('vector-difference', diff)
  # collect the final loss and print it to file
  sum_loss = tf.reduce_sum(loss_array.stack(), name="optimizee_sum_loss")
  if FLAGS.instrumentation:
    tf.summary.scalar("loss/final_loss", final_loss)
    tf.summary.scalar("loss/average_loss", sum_loss)

  # create reset op to be called at the begining of each new epoch
  with tf.name_scope("reset_loop_vars"):
    reset_variables = (nest.flatten(util.nested_variable(
        initial_states)) + variables + constants)
    reset_op = [tf.variables_initializer(reset_variables), loss_array.close()]

  # create update op to update the final vars
  with tf.name_scope("update"):
    update_op = (nest.flatten(
        util.nested_assign(variables, new_variables)) +
        nest.flatten(util.nested_assign(
                     util.nested_variable(initial_states),
                     new_states)))

  optimize_op = optimizer.optimize(sum_loss)

  return optimize_op, reset_op, update_op, final_loss, new_variables


def run_batch(sess, ops, batch_x, batch_y):
  """Runs all operation for a training batch
  In this case a training batch contains 20 normal batches,
  one for each unroll step in the loop
  :param sess: tensorflow session
  :param cost: final loss from unroll
  :param update: update variables after unroll op
  :param optimise: operation to reduce unroll op
  :param dic: feed_dict
  """
  start = timer()
  dic = {'x_inpt:0': batch_x, 'labels:0': batch_y}
  cost, upd, opt, summ = sess.run(ops, feed_dict=dic)
  return timer() - start, cost, summ


def run_epoch(sess, reset, ops):
  """ Runs an epoch with all dataset in batches
  :param sess: session
  :param reset: operation used to reset variables between epochs
  :param ops: operations to run
  :returns: tuplecontaining the time it took and the last cost
  """
  start = timer()
  x_train, y_train, _, _ = datasets.get_truncated_data()

  sess.run(reset)
  for _, data in enumerate(zip(x_train, y_train)):
    _, cost, summ = run_batch(sess, ops, data[0], data[1])
  # TODO: implement average cost and time and add it to summary
  return timer() - start, cost, summ


def main(_):
  """Runs the learning to learn ops for
  a number of epochs defined in flags
  """
  if FLAGS.seed:
    tf.set_random_seed(FLAGS.seed)

  x_inpt = tf.placeholder(dtype=tf.float32,
                          shape=[None, None, 784],
                          name='x_inpt')
  labels = tf.placeholder(dtype=tf.int32,
                          shape=[None, None, 10],
                          name='labels')

  # configure testing
  test_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 784])
  test_labels = tf.placeholder(dtype=tf.int32, shape=[None, 10])
  report = AccuracyReport()

  optimize_op, reset, update, final_loss, _ = unroll({
      'inputs': x_inpt,
      'labels': labels
  })

  merged_summary_op = tf.summary.merge_all()

  with tf.Session().as_default() as sess:
    sess.run(tf.global_variables_initializer())

    # prepare summaries
    final_dir = util.get_log_dir()
    graph = sess.graph if FLAGS.print_graph else None
    train_logger = Logger(FLAGS.logdir + '/train/' + final_dir, graph)
    test_logger = Logger(FLAGS.logdir + '/test/' + final_dir)
    if FLAGS.instrumentation and FLAGS.print_graph:
      print("[INFO] Session graph printed to file."
            " It can be viewed in Tensorboard.")
    # run training
    for epoch in range(FLAGS.num_epochs):
      time, cost, summary = run_epoch(sess, reset,
                                      [final_loss, update,
                                       optimize_op,
                                       merged_summary_op])
      train_logger.add_summary(summary, epoch+1)

      if FLAGS.instrumentation:
        print("[INFO] Epoch: {} \t Cost: {} \t Time {}".format(
            epoch+1, cost, time))

      if (epoch+1) % FLAGS.num_epochs_test == 0:
        # get model with latest updates and evaluate it
        with tf.name_scope(MODEL_CONSTANTS.FINAL_SCOPE):
          test_optimizee = OptimizeeFC()
          test_model = test_optimizee.build()
        clean_accuracy, adv_accuracy = evaluate(
            sess, test_inputs, test_labels, test_model, report)

        if FLAGS.instrumentation:
          print("[INFO] Test accuracy: \t Clean: {} \t Adversarial: {}".format(
              clean_accuracy, adv_accuracy))
          test_logger.log_scalar('/test/test_accuracy',
                                 clean_accuracy, epoch+1)
          test_logger.log_scalar(
              '/test/adversary_accuracy/' + FLAGS.test_attack, adv_accuracy, epoch+1)

  tf.reset_default_graph()


def evaluate(sess, test_inputs, test_labels, model, report):
  """Evaluates model against normal test examples and adversarial
  training examples
  :param sess: tensorflow session
  :param test_inputs: placeholder for test inputs
  :param test_labels: placeholder for test labels
  :param model: cleverhans model
  :param report: cleverhans report
  :returns: tuple containing accuracy on clean testing examples
      and accuracy on adversarial example
  """
  with tf.variable_scope("clean_evaluation"):
    x_train, y_train, x_test, y_test = datasets.get_clever_mnist()
    # test on test datase
    test_logits = model.get_logits(test_inputs)
    clean_accuracy = adv_util.evaluate(sess, report, test_inputs, test_labels,
                                       test_logits, x_test, y_test,
                                       'clean_train_clean_eval',
                                       MODEL_CONSTANTS.EVAL_PARAMS, False)
    # adversarial testing on train dataset
    attack_type, params = ADV_CONSTANTS.get_attack_details(FLAGS.test_attack)
    attack = attack_type(model, sess=sess)

  with tf.variable_scope("adversarial_evaluation"):
    adv_x = attack.generate(test_inputs, **params)
    adv_logits = model.get_logits(adv_x)
    adv_accuracy = adv_util.evaluate(sess, report, test_inputs, test_labels,
                                     adv_logits, x_train, y_train,
                                     'clean_train_adv_eval',
                                     MODEL_CONSTANTS.EVAL_PARAMS, True)
  return clean_accuracy, adv_accuracy


if __name__ == '__main__':
  tf.app.run()
