from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import unittest

import numpy as np


class CustomTest(unittest.TestCase):
    def setUp(self):
        self.test_start = time.time()
        # always use the same reandom seed
        # during tests
        np.random.seed(1234)
        print("Test: ", self.id, " started.")

    def tearDown(self):
        print("Test: ", self.id, "took ", time.time() -
              self.test_start, " seconds.")

    def assertClose(self, x, y, *args, **kwargs):
        assert np.allclose(x, y, *args, **kwargs), (x, y)
