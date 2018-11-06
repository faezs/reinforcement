from dqn import phi
import unittest
import numpy as np
import tensorflow as tf
import logging


class DQNTest(tf.test.TestCase):
    def testPhi(self):
        tf.logging.set_verbosity(logging.WARN)
        b = 64
        input_shape = (b, 4, 210, 160, 3)
        target_shape = (b, 84, 84, 4)
        images = np.random.uniform(low=0.0, high=255.0,
                                   size=input_shape)
        with self.test_session():
            small_squares = phi(images)
            actual_shape = small_squares.eval().shape
            self.assertEqual(actual_shape, target_shape)
