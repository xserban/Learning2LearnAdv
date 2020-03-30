"""Base class for the optimizee

The optimizee is based on cleverhans models
At the moment, the configurations for the internal
layers is maintained in this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import l2l.constants as ct
from l2l.adversarial.models.custom_model import CustomModel
from l2l import util


FLAGS = tf.flags
FLAGS = FLAGS.FLAGS

MODEL_CONSTANTS = ct.OptimizeeConstants()
ADVERSARIAL_CONST = ct.AdversarialConstants()


class OptimizeeFC():
  """A model for the optimizee network
  with only fully connected layers
  """

  def __init__(self):
    """Initializes an optimizee network
    """
    # init empty model
    self.model = None
    self.attack = None
    self.adv_params = None

  def build(self, input_shape=[None, 784], custom_variables=None,
            scope="optimizee_vars"):
    """Builds the computational graph for the optimziee
    :param input_shape: an array with the dimensions of the input
    :param custom_variables: an array with variables in the same
        order as the networks' variables - this array is used
        to build the network with these variables and disconnect
        the gradients in the graph.
    :returns: cleverhans model for this class
    """
    template = MODEL_CONSTANTS.get_fc_template()
    mlp_args = {'layers': template,
                'input_shape': input_shape,
                'scope': scope}
    if custom_variables is None:
      self.model = CustomModel(**mlp_args)
    else:
      self.model = util.make_with_custom_variables(
          CustomModel, mlp_args, custom_variables)

    return self.model

  def logits(self, inputs):
    """Builds the computational graph until the logits operation
    :param inputs: a tensor containing the inputs
    :param labels: a tensor containing the labels
    """
    if not self.model:
      self.build(input_shape=inputs.get_shape().as_list())

    return self.model.get_logits(inputs)

  def get_loss(self, inputs, labels):
    """Applies softmax cross entropy with logits and returns the mean
    :param inputs: a tensor containing a batch of data
    :param labels: a tensor containing the non-categorical labels
    :returns: an operation for reduce_mean
    """
    logits = self.logits(inputs)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits, name="cross_entropy")

    return tf.reduce_mean(loss)

  def loss(self, inputs, labels, custom_variables=None):
    """Builds the graph of the optimizee
    and returns the loss function given the input params
    :param inputs: a tensor containing the inputs
    :param labels: a tensor containing the labels
    :param custom_variables: an array with variables in the same
        order as the networks' variables - this array is used
        to build the network with these variables and disconnect
        the gradients in the graph.
    """
    if custom_variables is not None:
      self.build(inputs.get_shape().as_list(),
                 custom_variables=custom_variables)

    return self.get_loss(inputs, labels)

  def _set_attack(self, inputs=None):
    """Builds and sets self.attack
    :param inputs: a tensor with inputs used to get shape
    """
    if not self.model:
      if inputs is not None:
        self.build(inputs.get_shape().as_list())
      else:
        self.build()

    attack_type, self.adv_params = ADVERSARIAL_CONST.get_attack_details(
        FLAGS.train_attack)
    sess = tf.get_default_session()
    self.attack = attack_type(self.model, sess=sess)

  def generate_adv(self, inputs, labels=None):
    """Generates adversarial examples for a tensor with inputs
    :param inputs: a tensor with inputs
    :param labels: a tensor with labels - currently not used
      in order to avoid label leaking
    """
    if self.attack is None:
      self._set_attack(inputs)

    return self.attack.generate(inputs, **self.adv_params)
