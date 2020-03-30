"""
An extension of the MLPGet class (specific to this project) with
custom variable_scope
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from cleverhansl2l.picklable_model_get import MLPGet


class CustomModel(MLPGet):
  """Extends Cleverhans multi layer perceptron
  with reusable variable scope
  """

  def __init__(self, layers, input_shape, scope="optimizee_vars"):
    """Saves scope and calls the super init
    :param layers: template for layers
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
      # run template before
      lay = layers()
      super(CustomModel, self).__init__(lay, input_shape)

    self.scope = scope

    # self.fprop(tf.placeholder(tf.float32, [128, 784]))
    # Put a reference to the params in self so that the params get pickled
    # self.params = self.get_params()

  def fprop(self, x, **kwargs):
    """Extends forward propagation with
    variable scope
    """
    with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
      out = super(CustomModel, self).fprop(x, **kwargs)
    return out

  def get_layer_names(self):
    """Currently not implemented because it is not needed
    """
    raise NotImplementedError
