from dqn import phi, DQNAgent, ReplayMemory, linear_decay
import unittest
import numpy as np
import tensorflow as tf
from utils import combined_shape
import gym


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
    def get_agent(env, **kwargs):
       replay_capacity = 1e6
       n_episodes = 10e7

       return DQNAgent(env=env or gym.make('CartPole-v0'),
                       n_episodes=n_episodes,
                       replay_capacity=replay_capacity,
                       **kwargs)

    def testPhi(self):
        b = 64
        input_shape = (b, 4, 210, 160, 3)
        target_shape = (b, 84, 84, 4)
        images = self.get_rand(input_shape)
        with self.cached_session():
            small_squares = phi(images)
            actual_shape = small_squares.eval().shape
            self.assertEqual(actual_shape, target_shape)

    def testDNNThetaShape(self):
        b_size = 32
        agent = self.get_agent(None, arch='deepmind')
        out_dim = int(agent.env.action_space.n)

        # A Q-network maps input_shape -> [b_size, action_dim]
        input_shape = [b_size, 84, 84, 4]
        target_shape = [b_size, out_dim]
        sample_state = self.get_rand(input_shape, tensor=True)
        outputs = agent._theta(sample_state, name='test', reuse=False)
        self.assertEqual(outputs.get_shape().as_list(),
                         target_shape)

    def testReplayMemory(self):
        od = [84, 84, 4]
        ad = [8, 10]
        rd = [5]
        s = int(10000)
        b = 32

        rm = ReplayMemory(obs_dim=od,
                          act_dim=ad,
                          r_dim=rd,
                          size=s)
        o = self.get_rand(od)
        a = self.get_rand(ad)
        r = self.get_rand(rd)
        d = 0
        for _ in range(1000):
            rm.store(o, a, r, o, d)

        o_s, a_s, r_s, on_s, d_s = rm.sample(b)

        self.assertEqual(o_s.shape, combined_shape(b, od))
        self.assertEqual(a_s.shape, combined_shape(b, ad))
        self.assertEqual(r_s.shape, combined_shape(b, rd))
        self.assertEqual(on_s.shape, combined_shape(b, od))
        self.assertEqual(d_s.shape, combined_shape(b))

    def testE2E_Q(self):
        env = gym.make('CartPole-v0')
        agent = self.get_agent(env)
        #agent.learn()

    def test_linear_schedule(self):
        lsched = linear_decay(init=1.0, final=0.0, n_steps=10)
        init = 1
        for i in range(11):
            s = next(lsched)
        self.assertAlmostEqual(s, 0)
