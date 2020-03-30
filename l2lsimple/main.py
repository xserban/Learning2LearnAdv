from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from optimizee import Optimizee
from optimizer import Optimizer
from util import *

flags = tf.flags
logging = tf.logging

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_epochs", 2000, "Number of training epochs.")
flags.DEFINE_integer("training_steps", 20,
                     "Number of losses gathered before optimisation || optimisation steps per epoch.")
flags.DEFINE_integer("rnn_layers", 2, "Number of layers in the Optimizer RNN.")
flags.DEFINE_integer(
    "state_size", 20, "Number of units in an LSTM - at the moment is equal to the training steps.")
flags.DEFINE_integer("problem_dims", 10,
                     "Nr of variables for the problem to optimize.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_bool("instrumentation", True, "Show print/plots, etc.")
flags.DEFINE_bool("state_tuple", True,
                  "Pass a tuple with output/hidden state to LSTM.")


def learn(optimizee, optimizer):
  starting_point = tf.random_uniform([FLAGS.problem_dims], -1, 1)
  losses = []
  point = starting_point
  state = None

  for _ in range(FLAGS.training_steps):
    loss = optimizee(point)
    losses.append(loss)
    grads, = tf.gradients(loss, point)
    update, state = optimizer(grads, state)
    point += update

  return losses


def main():
  # create optimizer / optimizee and
  # comparison optimizers
  optimizer = Optimizer()
  optimizee = Optimizee()
  rnn = optimizer.inference()

  rnn_losses = learn(optimizee.evaluate, optimizer.evaluate)
  sum_losses = tf.reduce_sum(rnn_losses)

  apply_update = optimizer.optimize(sum_losses)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    average = 0
    for i in range(FLAGS.num_epochs):
      err, _ = sess.run([sum_losses, apply_update])
      average += err
      if i % 1000 == 0:
        if FLAGS.instrumentation:
          print("Training epoch {0}".format(i))
          print("Average error {0}".format(
              average/1000 if i != 0 else average))
          average = 0

    if FLAGS.instrumentation:
      print("Finished training! \n Average err {0}".format(average / 1000))


if __name__ == '__main__':
  main()
