from collections import namedtuple
import gym
import tensorflow as tf


Transition = namedtuple('Transition', ['phi_t',
                                       'a_t',
                                       'r_t',
                                       'phi_next'])

class DQNAgent(object):
    def __init__(self, env, n_episodes, replay_capacity, arch=None, action_space=None):
        self.env = env
        self.n_episodes = n_episodes
        self.replay_capacity = replay_capacity
        self.arch = arch or 'deepmind_cnn'
        self.history = []

    def init_replay_memory(self):
        self.replay_memory = []

    def phi(self, state):
        """
        Expects
        """
        return state

    def construct_Q(self, graph=None):
        if not graph:
            graph = tf.Graph()
        with graph.as_default() as g:
            self.construct_Q()
        return

    def eps_greedy_Q(self, x_t):
        q_outputs = self.Q.predict(x_t)


    def run_episode(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(self.env.action_space.sample())
        st = (obs, )
        phi = self.phi(st)
        while True:
            action = self.eps_greedy_Q(obs)
            obs_next, r, done, info = self.env.step(action)
            if done:
                break
            st_next = (st, action, obs_next)
            phi_next = self.phi(st_next)
            transition = (phi, action, r, phi_next)
            self.save_transition(transition)

            self.run_training_step(self.sample_transitions())
            obs = obs_next
            st = st_next
            phi = phi_next
        return

def phi(images,
        resize_to=(110, 84),
        crop_to=(84, 84),
        method=tf.image.ResizeMethod.BILINEAR):
    """
    Takes in a [b_size, 4, 210, 160, 3] images
    - converts each to grayscale:
      -> [b_size, 4, 210, 160, 1]
      -> squeeze: [b_size, 4, 210, 160]
      -> transpose: [b_size, 210, 160, 4]
    - downsamples each to [110 x 84]
    - crops each to 84 x 84
    - stacks them into [84, 84, 4] tensor
    """
    i = tf.image

    with tf.name_scope('phi', values=images) as scope:
        # [b_size, 4, 210, 160, 3] -> [b_size, 210, 160, 4]
        perm_grayscale = tf.transpose(tf.squeeze(
            i.rgb_to_grayscale(images,
                               name='rgb2grayscale'),
            axis=-1),
                                      perm=[0, 2, 3, 1])

        # [b_size, 210, 160, 4] -> [b_size, 110, 84, 4]
        downsample = i.resize_images(images=perm_grayscale,
                                     size=resize_to,
                                     method=method,
                                     align_corners=False,
                                     preserve_aspect_ratio=False)

        # [b_size, 110, 84, 4] -> [b_size, 84, 84, 4]
        square = i.resize_image_with_crop_or_pad(downsample,
                                                 target_height=crop_to[0],
                                                 target_width=crop_to[1])

        return square
