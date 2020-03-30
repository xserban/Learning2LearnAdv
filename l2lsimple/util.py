import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

flags = tf.flags
FLAGS = flags.FLAGS


def g_sgd(gradients, state, learning_rate=0.1):
  """ Stochastich gradient """
  return -learning_rate*gradients, state


def g_rms(gradients, state, learning_rate=0.1, decay_rate=0.99):
  """ RMSProp """
  if state is None:
    state = tf.zeros([DIMS])
  state = decay_rate*state + (1-decay_rate)*tf.pow(gradients, 2)
  update = -learning_rate*gradients/(tf.sqrt(state)+1e-5)
  return update, state


def plot(*args):
  """" draws several plots """
  hdls = []
  x_axis = np.arange(FLAGS.training_steps)

  plt.figure()
  for count, arg in enumerate(args):
    p = plt.plot(x_axis, arg[0], label=arg[1])
    hdls.append(p[0])

  plt.legend(handles=hdls)
  plt.title('losses')
  plt.draw()


def show_plot():
  """ Show plot """
  plt.show()
