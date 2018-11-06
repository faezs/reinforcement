from dqn import phi, DQNAgent
import unittest
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

class DQNTest(tf.test.TestCase):

    def tearDown(self):
        tf.reset_default_graph()

    @staticmethod
    def get_rand(size, tensor=False):
        r = np.random.uniform(low=0.0, high=255.0,
                              size=size)
        if not tensor:
            return r
        return tf.convert_to_tensor(r, dtype=tf.float32)

    @staticmethod
    def get_agent(env):
       action_dim = 4
       replay_capacity = 1e6
       n_episodes = 10e7

       return DQNAgent(env=env,
                       n_episodes=n_episodes,
                       replay_capacity=replay_capacity,
                       action_dim=action_dim)

    def testPhi(self):
        b = 64
        input_shape = (b, 4, 210, 160, 3)
        target_shape = (b, 84, 84, 4)
        images = self.get_rand(input_shape)
        with self.cached_session():
            small_squares = phi(images)
            actual_shape = small_squares.eval().shape
            self.assertEqual(actual_shape, target_shape)

    def testQNetworkShape(self):
        env = None
        b_size = 32
        agent = self.get_agent(env)

        # A Q-network maps input_shape -> [b_size, action_dim]
        input_shape = [b_size, 84, 84, 4]
        target_shape = [b_size, agent.action_dim]
        sample_state = self.get_rand(input_shape, tensor=True)
        outputs = agent._construct_Q(sample_state, name='test', reuse=False)
        self.assertEqual(outputs.get_shape().as_list(),
                         target_shape)
