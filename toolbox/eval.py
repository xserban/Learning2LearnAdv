"""Some methods from the toolbox used for
evaluation purposes
"""
import math
import tensorflow as tf

FLAGS = tf.flags
FLAGS = FLAGS.FLAGS


def compute_accuracy(sess, x_phd, y_phd, x_test, y_test, optimizee):
  """Computes the accuracy for an operation, given the input
  and label placeholders, the data and the operation
  :param sess: session
  :param x_phd: input placeholder
  :param y_phd: labels placeholder
  :param x_test: testing data
  :param y_test: testing labels
  :param optimizee: optimizee to call for logits
  :returns:
  """
  accuracy = 0
  num_batches = int(math.ceil(float(len(x_test)) / FLAGS.batch_size))
  for batch in range(num_batches):
    start = batch * FLAGS.batch_size
    end = min(len(x_test), start + FLAGS.batch_size)
    batch_x = x_test[start:end]
    batch_y = y_test[start:end]

    logits = optimizee.logits(x_phd)
    correct = tf.equal(tf.argmax(logits, axis=-1),
                       tf.argmax(y_phd, axis=-1))
    acc = tf.reduce_sum(tf.cast(correct, tf.int32))

    accuracy += sess.run(acc,
                         feed_dict={
                             x_phd: batch_x,
                             y_phd: batch_y})
  return accuracy
