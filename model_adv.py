from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from cleverhans.model import Model


class Model_adv(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, CNN):
        super(Model_adv, self).__init__()

        self.layer_names = []
        self.layers = []

        graph = tf.get_default_graph()

        self.layers.append(graph.get_tensor_by_name("logits/BiasAdd:0"))
        self.layer_names.append("logits")

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))


