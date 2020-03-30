"""This module defines any utility
methods needed only by the adversarial module
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from cleverhansl2l.utils_tf import model_eval

from l2l.optimizee_fc import OptimizeeFC


def attack(att, inputs, params):
  """Generate attack for a sample -
  This method was added in order to apply any
  pre-processing steps to the samples
  :param att: operation specific for attack
  :param inputs: sample
  :param params: fgsm params
  :returns generate operation:
  """
  return att.generate(inputs, **params)


def evaluate(sess, report, inputs, labels,  predictions,
             x_set, y_set, report_key, eval_args,
             is_adv=None):
  """Prints the results of the evaluation and returns the accuracy
  :param sess: tensorflow session
  :param report:
  :param inputs:
  :param labels:
  :param predictions:
  :param x_set:
  :param y_set:
  :param report_key:
  :param eval_params:
  :param is_adv:
  :returns tensorflow operation:
  """
  accuracy = model_eval(sess, inputs, labels, predictions,
                        x_set, y_set, args=eval_args)
  setattr(report, report_key, accuracy)
  if is_adv is None:
    report_text = None
  elif is_adv:
    report_text = 'adversarial'
  else:
    report_text = 'legitimate'
  if report_text:
    print('Test accuracy on %s examples: %0.4f' % (report_text, accuracy))

  return accuracy


def generate_fgsm(custom_vars, args, eps=0.3, clip_min=0.0, clip_max=1.0):
  """Generates adversarial examples for all inputs
  :param inputs: a tensor containing the inputs
  :param gradients: a tensor containing gradients of the loss
      w.r.t inputs
  :param eps: step size
  :param clip_min: minimum value/pixel
  :param clip_max: maximum value/pixel
  :returns: operation clip_by_value from cleverhansl2l
  """
  with tf.variable_scope("adversarial"):
    optimizee = OptimizeeFC()
    loss = optimizee.loss(custom_variables=custom_vars, **args)
    grads = tf.gradients(loss, args['inputs'])
    sign = tf.math.sign(grads[0])
    pert = tf.stop_gradient(sign)
    scaled = pert * 0.0
    adv = args['inputs'] + scaled
    # TODO: Clip by value
    return adv
