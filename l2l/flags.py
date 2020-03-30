"""This module contains all default configuration flags
for running learning-to-learn
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags

# FLAGS SPECIFIC TO META-LEARNING
FLAGS.DEFINE_string("input_shape", "-1, 784", "Input size.")
FLAGS.DEFINE_integer("batch_size", 128, "Batch size.")

# FLAGS SPECIFIC TO TRAINING AND INPUTS
FLAGS.DEFINE_integer(
    "num_units", 100, "Number of hidden units in fully connected layer.")
FLAGS.DEFINE_integer("num_classes", 10, "Number of output classes.")
FLAGS.DEFINE_float("learning_rate", 0.01, "Learning rate.")
FLAGS.DEFINE_integer("num_epochs", 12, "Number of training epochs.")
FLAGS.DEFINE_integer("num_epochs_test", 3,
                     "Number of training epochs for testing purposes.")

# FLAGS SPECIFIC TO OPTIMIZER RNN
FLAGS.DEFINE_integer(
    "rnn_layers", 2, "Numbers of layers in the Optimizer RNN.")
FLAGS.DEFINE_integer("state_size", 20, "Number of units in a LSTM.")
FLAGS.DEFINE_integer("truncated_backprop", 20,
                     "Number of losses gathered before optimisation.")
FLAGS.DEFINE_bool("state_tuple", True,
                  "Pass a tuple with output/hidden state to LSTM.")
FLAGS.DEFINE_bool("second_derivatives", False,
                  "Apply second derivatives when optimizing"
                  " aka link gradients from optimizee to the"
                  " optimizer graph (seee picture in original"
                  " paper).")

# FLAGS SPECIFIC TO ADVERSARIAL TRAINING
FLAGS.DEFINE_bool("adversarial", True,
                  "Train with adversarial examples")
FLAGS.DEFINE_bool("clipped", False, "Clip adversarial examples.")
FLAGS.DEFINE_bool("clean_train", True, "Train the model on clean examples?")
FLAGS.DEFINE_string("model_type", "linear",
                    "Which model to test adversarial examples on:"
                    " linear or convolutional")
FLAGS.DEFINE_integer("num_adv_units", 20, "Number of adversarial units ")
FLAGS.DEFINE_float("label_smoothing", 0.0, "Label smoothing percentage.")
FLAGS.DEFINE_integer("num_epochs_adv", 6,
                     "Number of epochs for testing adversarial training.")
FLAGS.DEFINE_integer("num_filters", 16, "Number of filters in CNN")
FLAGS.DEFINE_string("test_attack", "fgsm",
                    "Type of attack from cleverhans used during testing")
FLAGS.DEFINE_string("train_attack", "fgsm",
                    "Type of attack from cleverhans used for adv-training")

# FLAGS SPECIFIC TO ENVIRONMENT
FLAGS.DEFINE_integer("test_step", 1, "Number of epochs at which the "
                     "evaluation operation is called")
FLAGS.DEFINE_bool("print_graph", False, "Print session graph to tensorboard")

# FLAGS SPECIFIC TO REPRODUCIBILITY
FLAGS.DEFINE_integer("seed", 1234, "Seed")

# FLAGS SPECIFIC TO INSTRUMENTATION
FLAGS.DEFINE_bool("instrumentation", True, "Show prints/plots, etc.")
FLAGS.DEFINE_string("logdir", "./logs", "Logs directory")
FLAGS.DEFINE_bool("print_grads", True,
                  "Show gradient information in tensorboard.")

# FLAGS SPECIFIC TO PREPROCESSING
FLAGS.DEFINE_string("preprocessing", "multiply",
                    "preprocessing method")  # log_sign or multiply
FLAGS.DEFINE_integer("k", 20, "Preprocessing Constant")
FLAGS.DEFINE_float("optimizer_scale", 0.01,
                   "Scaling factor for optimizer output")


def set_flags(flags):
  """Deletes all flags and sets new ones
  :param flags: a dictionary with flags + values
  Some examples are in experiment/design.json
  """
  tf_flags = FLAGS.FLAGS

  for key, value in flags.items():
    tf_flags.__delattr__(key)
    define_flag = getattr(FLAGS, "DEFINE_"+value["type"])
    define_flag(key, value["value"], " ")
