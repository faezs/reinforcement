from collections import namedtuple
import gym
import tensorflow as tf


Transition = namedtuple('Transition', ['phi_t',
                                       'a_t',
                                       'r_t',
                                       'phi_next'])

class DQNAgent(object):
    def __init__(self, env, n_episodes, replay_capacity, arch=None, action_space):
        self.env = env
        self.n_episodes = n_episodes
        self.replay_capacity = replay_capacity
        self.arch = arch or 'deepmind_cnn'
        self.history = []

    def init_replay_memory(self):
        self.replay_memory = 
    
    def phi(self, state):
        """
        Expects 
        """
        return state

    def construct_Q(self):
