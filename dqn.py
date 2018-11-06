from collections import namedtuple
import gym
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

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


def conv2d(inputs, name,
           n_filters, kernel_size, strides,
           activation=tf.nn.relu, reuse=False):
    return tf.layers.conv2d(inputs,
                            filters=n_filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='valid',
                            activation=activation,
                            name=name,
                            reuse=reuse)

def fc(inputs, name, units, activation=None, reuse=False):
    return tf.layers.dense(inputs,
                           units=units,
                           activation=activation,
                           name=name,
                           reuse=reuse)

def flatten(inputs, name):
    return tf.layers.flatten(inputs, name=name)

# To use these with the functions in this lib,
# add the input tensor and a reuse flag
# in the construction context

Conv2DSpec = namedtuple('Conv2DSpec', ['name',
                                       'n_filters',
                                       'kernel_size',
                                       'strides',
                                       'activation'])

FCSpec = namedtuple('FCSpec', ['name',
                               'units',
                               'activation'])

FlattenSpec = namedtuple('FlattenSpec', ['name'])

SPEC_TO_LAYER = {Conv2DSpec: conv2d,
                 FCSpec: fc,
                 FlattenSpec: flatten}

def make_layer(inp, spec, reuse=False):
    ltype = type(spec)
    if not ltype in SPEC_TO_LAYER:
        raise TypeError('{} not found in layer library: {}'.format(ltype, SPEC_TO_LAYER.keys()))
    l_fn = SPEC_TO_LAYER[ltype]
    l_args = spec._asdict()
    l_args['inputs'] = inp
    if reuse:
        l_args['reuse'] = reuse
    return l_fn(**l_args)

deepmind_arch = (Conv2DSpec(name='conv1',
                            n_filters=16,
                            kernel_size=(16, 16),
                            strides=(4, 4),
                            activation=tf.nn.relu),
                 Conv2DSpec(name='conv2',
                            n_filters=32,
                            kernel_size=(4, 4),
                            strides=(2, 2),
                            activation=tf.nn.relu),
                 FlattenSpec(name='flatten_conv2'),
                 FCSpec(name='fc1',
                        units=256,
                        activation=tf.nn.relu))

Transition = namedtuple('Transition', ['phi_t',
                                       'a_t',
                                       'r_t',
                                       'phi_next'])

class DQNAgent(object):
    def __init__(self, env, n_episodes,
                 replay_capacity,
                 action_dim,
                 arch=deepmind_arch):
        self.env = env
        self.n_episodes = n_episodes
        self.replay_capacity = replay_capacity
        self.arch = arch
        self.action_dim = action_dim
        self.history = []

    def _init_replay_memory(self):
        self.replay_memory = []

    def _phi(self, state):
        """
        Expects
        """
        return state

    def _output_layer(self, inp, reuse=False):
        s = FCSpec(name='output',
                   units=self.action_dim,
                   activation=None)
        return make_layer(inp, s, reuse=reuse)

    def _construct_Q(self, inputs, name, reuse=False):
        x = inputs
        with tf.name_scope(name):
            for spec in self.arch:
                x = make_layer(x, spec, reuse)
            output = self._output_layer(x, reuse=reuse)
        return output

    def _dq_loss(self):
        return


    def _training_op(self):
        return

    def _eps_greedy_Q(self, x_t):
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
