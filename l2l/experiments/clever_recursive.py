from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from cleverhansl2l.model import Model
from cleverhansl2l.picklable_model import MLP, Linear, ReLU
from cleverhansl2l.picklable_model import Softmax
from tensorflow.contrib.learn.python.learn import monitored_session as ms

from l2l import util
from l2l import datasets

flags = tf.flags
FLAGS = flags.FLAGS


layers = [Linear(FLAGS.num_units, scope="linear_1"),
          ReLU(),
          Linear(FLAGS.num_classes, scope="logits"),
          ]
input_shape = (None, 784)

op = MLP
opkwargs = {'layers': layers, 'input_shape': input_shape}

batch_x, batch_y = datasets.get_clever_batch()


def loop_body(t, variables, x, y, fx_array):
  with tf.name_scope("loop_function"):
    model = util.make_with_custom_variables(op, opkwargs, variables)
    logits = model.get_logits(x)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=logits)
    fx_array = fx_array.write(t, loss)

  with tf.name_scope("time_step"):
    t_next = t+1

  next_variables = [x+0.01 for x in variables]
  return t_next, next_variables, x, y, fx_array


def create_tf_loop():
  variables, _ = util.get_variables(op, opkwargs)

  fx_array = tf.TensorArray(tf.float32, size=FLAGS.truncated_backprop + 1,
                            clear_after_read=False)

  _, x_final, _, _, fx_array = tf.while_loop(
      cond=lambda t, *_: t < FLAGS.truncated_backprop+1,
      body=loop_body,
      loop_vars=(0, variables, batch_x, batch_y, fx_array),
      parallel_iterations=1,
      swap_memory=True,
      name="unroll"
  )

  size = fx_array.size()
  return tf.reduce_sum(fx_array.stack(), name="loss"), size


def run_tf_loop():
  fx_array = create_tf_loop()

  with ms.MonitoredSession() as sess:
    tf.get_default_graph().finalize()

    loss, size = sess.run(fx_array)
    return loss


def create_loop_model():
  model = MLP(layers, input_shape)
