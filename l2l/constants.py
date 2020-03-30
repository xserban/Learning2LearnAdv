"""Class that holds all constants
"""
import tensorflow as tf

from cleverhansl2l.picklable_model_get import Linear, ReLU, Softmax, Conv2D, Flatten
from cleverhansl2l.attacks import FastGradientMethod
from cleverhansl2l.attacks import MadryEtAl

from l2l.singleton import Singleton

from l2l import flags

flags = tf.flags
FLAGS = flags.FLAGS


class OptimizeeConstants(metaclass=Singleton):
  """Holds all constants for the optimizer/trained model"""

  FINAL_SCOPE = 'trained_optimizee'

  CONV_LAYERS = [
      Conv2D(FLAGS.num_filters, (8, 8), (2, 2), "SAME"),
      ReLU(),
      Conv2D(FLAGS.num_filters * 2, (6, 6), (2, 2), "VALID"),
      ReLU(),
      Conv2D(FLAGS.num_filters * 2, (5, 5), (1, 1), "VALID"),
      ReLU(),
      Flatten(),
      Linear(FLAGS.num_classes, scope="logits"),
  ]

  def get_fc_template(self):
    def get_layers():
      return [Linear(20, scope="linear_1"),
              ReLU(),
              # Linear(FLAGS.num_units, scope="linear_2"),
              # ReLU(),
              # Linear(FLAGS.num_units, scope="linear_3"),
              # ReLU(),
              # Linear(FLAGS.num_units, scope="linear_4"),
              # ReLU(),
              # Linear(FLAGS.num_units, scope="linear_5"),
              # ReLU(),
              Linear(FLAGS.num_classes, scope="logits")]
    return tf.make_template('clever_model', get_layers, create_scope_now_=False)

  TRAIN_PARAMS = {
      'nb_epochs': FLAGS.num_epochs_adv,
      'batch_size': FLAGS.batch_size,
      'learning_rate': FLAGS.learning_rate
  }

  EVAL_PARAMS = {
      'batch_size': FLAGS.batch_size
  }


class AdversarialConstants(metaclass=Singleton):
  """Holds all constants for adversarial attacks"""
  FGSM_PARAMS = {
      'eps': 0.3,

  }
  MADRY_PARAMS = {
      'eps': 0.3,
      'nb_iter': 40,
      'eps_iter': 0.01
  }

  def add_clip(self, params):
    """Adds clipping attributes to attack params
    :param params: a dictionary with attack params
    """
    if FLAGS.clipped:
      params['clip_min'] = 0.
      params['clip_max'] = 1.
    return params

  def get_attack_details(self, attack):
    """Returns configuration for the attack studied
    The default configuration is for FGSM
    :param attack: string with attack type
    """
    if attack is 'fgsm':
      return FastGradientMethod, self.add_clip(self.FGSM_PARAMS)
    elif attack is 'madry':
      return MadryEtAl, self.add_clip(self.MADRY_PARAMS)
    else:
      return FastGradientMethod, self.add_clip(self.FGSM_PARAMS)

  def get_attack_detail(self, attack, detail):
    """Returns only one key for the attack
    :param detail: key
    :returns: string or numerical value
    """
    _, params = self.get_attack_details(attack)
    try:
      return str(params[detail])
    except:
      return ""

  def set_attack_params(self, key, values):
    """Sets the parameters of an attack. This class is a singleton
    so it only has one instance, therefore setting the params here
    will be propagated everywhere.
    :param key: attack
    :param value: values
    """
    setattr(self, key, values)
