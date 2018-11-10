from collections import namedtuple
import gym
import tensorflow as tf
import numpy as np
from utils import combined_shape, placeholder, placeholder_from_space


Transition = namedtuple('Transition', ['obs_t',
                                       'act_t',
                                       'r_t',
                                       'obs_tn',
                                       'done'])

ActFeed = namedtuple('ActFeed', ['phi_t',
                                 'stochastic',
                                 'update_eps'])


class ReplayMemory(object):
    def __init__(self, obs_dim, act_dim, r_dim, size):
        self.obs_buf = np.zeros(shape=combined_shape(size, obs_dim))
        self.act_buf = np.zeros(shape=combined_shape(size, act_dim))
        self.r_buf = np.zeros(shape=combined_shape(size, r_dim))
        self.obs_tn_buf = np.zeros(shape=combined_shape(size, obs_dim))
        self.done_buf = np.zeros(shape=combined_shape(size), dtype=np.float32)
        self.idx = 0
        self.max_idx = size

    def store(self, obs_t, act_t, r_t, obs_tn, done):
        if not self.idx < self.max_idx:
            raise Exception('Memory Capacity Exceeded')
        self.obs_buf[self.idx] = obs_t
        self.act_buf[self.idx] = act_t
        self.r_buf[self.idx] = r_t
        self.obs_tn_buf[self.idx] = obs_tn
        self.done_buf[self.idx] = done
        self.idx += 1

    def sample(self, batch_size=32):
        sample_idxs = np.random.choice(self.idx,
                                       batch_size,
                                       replace=True)
        obs_mb =self.obs_buf[sample_idxs, :]
        act_mb =self.act_buf[sample_idxs, :]
        r_mb = self.r_buf[sample_idxs, :]
        obs_tn_mb = self.obs_tn_buf[sample_idxs, :]
        done_mb = self.done_buf[sample_idxs]
        return Transition(obs_t=obs_mb,
                          act_t=np.squeeze(act_mb),
                          r_t=np.squeeze(r_mb),
                          obs_tn=obs_tn_mb,
                          done=np.squeeze(done_mb))


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

dnn_arch = (FCSpec(name='fc1',
                   units=256,
                   activation=tf.nn.relu),
            FCSpec(name='fc2',
                   units=256,
                   activation=tf.nn.relu),
            FCSpec(name='fc3',
                   units=256,
                   activation=tf.nn.relu))


archs = {'dnn':dnn_arch,
         'deepmind':deepmind_arch}


class DQNAgent(object):
    def __init__(self,
                 env,
                 n_episodes,
                 replay_capacity,
                 arch='dnn'):
        if not arch in archs:
            raise TypeError('{} not in registered architectures'.format(arch))
        self.env = env
        self.n_episodes = n_episodes
        self.replay_capacity = int(replay_capacity)
        self.arch = archs[arch]
        self.history = []
        self.batch_size = 32
        self.num_actions = int(self.env.action_space.n)

    def _init_replay_memory(self):
        self._buffer = ReplayMemory(self.env.observation_space.shape, 1, 1, size=self.replay_capacity)

    def _output_layer(self, inp, reuse=False):
        s = FCSpec(name='output',
                   units=self.env.action_space.n,
                   activation=None)
        return make_layer(inp, s, reuse=reuse)

    def _theta(self, inputs, name, reuse=False):
        x = inputs
        with tf.variable_scope(name):
            for spec in self.arch:
                x = make_layer(x, spec, reuse)
            output = self._output_layer(x, reuse=reuse)
        return output

    def _transition_phs(self):
        return Transition(
            obs_t=placeholder_from_space(self.env.observation_space),
            act_t=tf.placeholder(shape=[None], dtype=tf.int32),
            r_t=placeholder(dim=None),
            obs_tn=placeholder_from_space(self.env.observation_space),
            done=placeholder(None))

    def _act_phs(self):
        return ActFeed(
            phi_t=placeholder_from_space(self.env.observation_space),
            stochastic=tf.placeholder(tf.bool, (), name='stochastic'),
            update_eps=tf.placeholder(tf.float32, (), name='update_eps'))

    def _make_feed(self, phs, vals):
        # phs and vals must be the same kind of namedtuple
        assert type(phs) == type(vals), (phs, vals)
        # phs: {str, tf.placeholders}
        # vals: {str, np.array}
        feeds = {}
        for ph, val in zip(phs, vals):
            feeds[ph] = val
        return feeds

    def _act(self):
        phi_t, stochastic_ph, update_eps_ph = phs = self._act_phs()

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_t = self._theta(phi_t, 'theta_t')
        deterministic_actions = tf.argmax(q_t, axis=1)

        batch_size = tf.shape(phi_t)[0]

        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=self.num_actions, dtype=tf.int64)

        choose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps

        stochastic_actions = tf.where(choose_random, random_actions, deterministic_actions)

        eps_greedy_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda:deterministic_actions)

        update_eps_op = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        return phs, eps_greedy_actions, update_eps_op


    def _train_and_update_ops(self, loss, theta_t, theta_tn):
        opt = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train = opt.minimize(loss, var_list=theta_t)

        update_target_ops = []
        for var, var_target in zip(sorted(theta_t, key=lambda v: v.name),
                                   sorted(theta_tn, key=lambda v: v.name)):
            update_target_ops.append(var_target.assign(var))

        update_target_op = tf.group(*update_target_ops)
        return train, update_target_op

    def _build_deepQ_graph(self, gamma=1.0):

        # Add Action Ops
        act_feeds, eps_greedy_actions, eps_update = self._act()

        # Add Transition Placeholders
        phi_t, a_t, r_t, phi_tn, done = phs = self._transition_phs()

        # Add action value function for transition initial state (reusing self._act weights)
        q_t = self._theta(phi_t, 'theta_t', reuse=True)
        q_t_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='theta_t')

        # Add action value function for transition terminal state
        # (no weight reuse since the target network doesn't exist in the _act graph)
        q_tn = self._theta(phi_tn, 'theta_tn')
        q_tn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='theta_tn')

        # Scores for initial selected actions in these transitions
        y_i = tf.reduce_sum(q_t * tf.one_hot(a_t, self.num_actions), 1)

        # max_a_Q* i.e RHS of Bellman's Equation
        q_tn_best = tf.reduce_max(q_tn, 1)
        q_tn_best_masked = (1.0 - done) * q_tn_best
        q_tn_action_reward = r_t + gamma * q_tn_best_masked

        # Loss function derived from Bellman's Equation
        loss = tf.square(y_i - q_tn_action_reward)

        # SGD op on Loss function applied to q_function_act
        # Update op for setting q_function_train weights to q_function_act weights
        # (which are the ones actually optimized by SGD)
        train_qn, q_tn_update_op = self._train_and_update_ops(loss, q_t_vars, q_tn_vars)

        return act_feeds, eps_greedy_actions, eps_update, phs, train_qn, q_tn_update_op

    def k_skip_sample(self, a):
        samples = zip(*(self.env.sample(a) for _ in range(self.skip_k)))
        obs_tn, r_t, dones, info = samples
        #return np.stack(obs_tn, axis=0), r_t = np.st


    def learn(self, n_episodes=100000, start_training=32, batch_size=32, update_qtn_every=64):
        self._init_replay_memory()

        self.env.reset()

        # initialize eps decay generator
        eps_val = linear_decay(init=1, final=0.1, n_steps=1000)

        with tf.Session() as session:
            # Build e-greedy policy and q-networks training graph
            act_phs, eps_greedy_actions, eps_update_op, transition_phs, train_qn_op, update_qtn_op = self._build_deepQ_graph()

            # Initialize tf variables
            # set qtn weights to be equal to qt weights
            session.run(tf.global_variables_initializer())
            session.run(update_qtn_op)

            eps_reward = []
            for i in range(n_episodes):
                self.env.reset()
                done = False
                stochastic = True
                obs_t  = self.env.reset()
                t = 0
                eps_reward.append(0.0)
                while not done:
                    # 1) Get Action for current observation
                    a_t, _ = session.run(fetches=[eps_greedy_actions, eps_update_op],
                                      feed_dict=self._make_feed(act_phs, ActFeed(phi_t=np.array(obs_t)[None],
                                                                                 stochastic=stochastic,
                                                                                 update_eps=next(eps_val))))

                    # From Array to Scalar
                    a_t = a_t[0]
                    obs_tn, r_t, done, info = self.env.step(a_t)

                    eps_reward[i] += r_t
                    #print('episode: {}, timestep: {}, action: {}, cumulative_reward: {}'.format(i, t, a_t, eps_reward[i]))
                    self.env.render()

                    self._buffer.store(obs_t=obs_t, act_t=a_t, r_t=r_t, obs_tn=obs_tn, done=float(done))

                    if t > start_training:
                        _ = session.run(fetches=[train_qn_op],
                                        feed_dict=self._make_feed(transition_phs,
                                                                  self._buffer.sample(batch_size)))
                    if t % update_qtn_every:
                        _ = session.run(fetches=[train_qn_op, update_qtn_op],
                                        feed_dict=self._make_feed(transition_phs,
                                                                  self._buffer.sample(batch_size)))

                    obs_t = obs_tn
                    t += 1
                print('episode: {}, timesteps: {}, cumulative_reward: {}'.format(i, t, eps_reward[i]))


def linear_decay(init, final, n_steps):
    delta = (init - final) / n_steps
    assert init > final, 'init: {} must be greater than final: {}'.format(init, final)
    counter = init
    for i in range(n_steps):
        yield counter
        counter = counter - delta
    while True:
        yield counter

def get_agent(env, **kwargs):
    replay_capacity = 1e6
    n_episodes = 10e7

    return DQNAgent(env=env or gym.make('CartPole-v0'),
                    n_episodes=n_episodes,
                    replay_capacity=replay_capacity,
                    **kwargs)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = get_agent(env)
    agent.learn()
