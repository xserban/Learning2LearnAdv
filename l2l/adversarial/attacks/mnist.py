"""This module applies adversarial attacks on
MNIST
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf

from cleverhansl2l.attacks import FastGradientMethod
from cleverhansl2l.loss import CrossEntropy
from cleverhansl2l.dataset import MNIST
from cleverhansl2l.train import train
from cleverhansl2l.utils import AccuracyReport

from l2l import flags
from l2l.adversarial.models.custom_model import CustomModel
from l2l.adversarial import util as utl
from l2l.constants import *

FLAGS = tf.flags
FLAGS = FLAGS.FLAGS

MODEL_CONSTANTS = OptimizeeConstants()
ADV_CONSTANTS = AdversarialConstants()

model_params = {
    'input_shape': None,
    'layers': None
}

report = AccuracyReport()
tf.set_random_seed(1234)
rng = np.random.RandomState([2017, 8, 30])
mnist = MNIST(train_start=0, train_end=60000,
              test_start=0, test_end=10000)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')


def get_model(model_type, scope):
  """Returns a cleverhans model class
  :param model_type: linear or cnn model
  :param kwargs: model arguments
  :returns a class:
  """
  if model_type == 'linear':
    model_params['layers'] = MODEL_CONSTANTS.get_fc_template()
    model_params['input_shape'] = [None, 784]
    global x_train, x_test
    x_train = np.reshape(x_train, [-1, 784])
    x_test = np.reshape(x_test, [-1, 784])
    return CustomModel(scope=scope, **model_params)
  elif model_type == 'cnn':
    model_params['layers'] = MODEL_CONSTANTS.CONV_LAYERS
    model_params['input_shape'] = [-1, 28, 28, 1]
    return CustomModel(scope=scope, **model_params)
  else:
    raise TypeError('Model type not recognised')


def train_clean(sess, inputs, labels):
  """Train without adversarial examples
  """
  def _evaluate():
    """Interface o evaluation method for
    cleverhans train
    """
    utl.evaluate(sess, report, inputs, labels, preds,
                 x_test, y_test,
                 'clean_train_clean_eval', MODEL_CONSTANTS.EVAL_PARAMS, False)

  if FLAGS.clean_train:
    model = get_model(FLAGS.model_type, 'train_1')
    preds = model.get_logits(inputs)
    loss = CrossEntropy(model, smoothing=FLAGS.label_smoothing)

  train(sess, loss, x_train, y_train,
        evaluate=_evaluate,
        args=MODEL_CONSTANTS.TRAIN_PARAMS, rng=rng, var_list=model.get_params())

  # evaluate model on adversarial examples
  fgsm = FastGradientMethod(model, sess=sess)
  adv_x = fgsm.generate(
      inputs, **ADV_CONSTANTS.get_attack_details(FLAGS.train_attack)[1])
  preds_adv = model.get_logits(adv_x)

  utl.evaluate(sess, report, inputs, labels, preds_adv, x_test,
               y_test, 'clean_train_adv_eval', MODEL_CONSTANTS.EVAL_PARAMS, True)

  # calculate training error
  utl.evaluate(sess, report, inputs, labels, preds, x_train,
               y_train, 'train_clean_train_clean_eval', MODEL_CONSTANTS.EVAL_PARAMS)


def adv_training(sess, inputs, labels):
  """Train and evaluates a model on adversarial examples
  """
  adversarial_model = get_model(FLAGS.model_type, 'train_adv')
  fgsm2 = FastGradientMethod(adversarial_model, sess=sess)

  def attack(inputs):
    return fgsm2.generate(inputs, **ADV_CONSTANTS.get_attack_details(FLAGS.train_attack)[1])

  loss2 = CrossEntropy(
      adversarial_model, smoothing=FLAGS.label_smoothing, attack=attack)
  adv_x2 = attack(inputs)

  preds_2 = adversarial_model.get_logits(inputs)
  preds2_adv = adversarial_model.get_logits(adv_x2)

  def evaluate2():
    """Interface to cleverhans train
    """
    # Accuracy of adversarially trained model on legitimate test inputs
    utl.evaluate(sess, report, inputs, labels, preds_2, x_test, y_test,
                 'adv_train_clean_eval', MODEL_CONSTANTS.EVAL_PARAMS, False)
    # Accuracy of the adversarially trained model on adversarial examples
    utl.evaluate(sess, report, inputs, labels, preds2_adv, x_test,
                 y_test, 'adv_train_adv_eval', MODEL_CONSTANTS.EVAL_PARAMS, True)

  # Perform and evaluate adversarial training
  train(sess, loss2, x_train, y_train, evaluate=evaluate2,
        args=MODEL_CONSTANTS.TRAIN_PARAMS, rng=rng, var_list=adversarial_model.get_params())

  # Calculate training errors
  utl.evaluate(sess, report,  inputs, labels, preds2_adv, x_train,
               y_train, 'train_adv_train_clean_eval', MODEL_CONSTANTS.EVAL_PARAMS)
  utl.evaluate(sess, report, inputs, labels, preds2_adv, x_train,
               y_train, 'train_adv_train_adv_eval', MODEL_CONSTANTS.EVAL_PARAMS)


def main(argv=None):
  """Trains a model
  """
  shape = [None, 784] if FLAGS.model_type is 'linear' else [None, 28, 28, 1]
  inputs = tf.placeholder(tf.float32, shape=shape)
  labels = tf.placeholder(tf.float32, shape=[None, 10])

  init = tf.global_variables_initializer()
  with tf.Session() as sess:
    sess.run(init)

    train_clean(sess, inputs, labels)
    print('Repeating the process, using adversarial training')
    adv_training(sess, inputs, labels)


if __name__ == '__main__':
  tf.app.run()
