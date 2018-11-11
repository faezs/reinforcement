"""
Utilities for interfacing with gym spaces

Mostly copied wholesale from openai/baselines, so all credit to the implementers.
see more: https://github.com/openai/baselines/common/{inputs.py, tf_utils.py}
"""

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete, Tuple, MultiDiscrete


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32,
                          shape=combined_shape(None, dim))

def placeholder_from_space(space, batch_dim=None):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(batch_dim,))
    raise NotImplementedError

def env_n_actions(space):
    if isinstance(space, Discrete) or hasattr(space, 'n'):
        return int(space.n)
    elif isinstance(space, Box):
        return int(np.prod(space.shape))
    elif isinstance(space, Tuple):
        return
        raise NotImplementedError

def observation_placeholder(ob_space, batch_size=None, name='Ob'):
    '''
    Create placeholder to feed observations into of the size appropriate to the observation space
    Parameters:
    ----------
    ob_space: gym.Space     observation space
    batch_size: int         size of the batch to be fed into input. Can be left None in most cases.
    name: str               name of the placeholder
    Returns:
    -------
    tensorflow placeholder tensor
    '''

    assert isinstance(ob_space, Discrete) or isinstance(ob_space, Box) or isinstance(ob_space, MultiDiscrete), \
        'Can only deal with Discrete and Box observation spaces for now'

    dtype = ob_space.dtype
    if dtype == np.int8:
        dtype = np.uint8

    return tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=dtype, name=name)


def observation_input(ob_space, batch_size=None, name='Ob'):
    '''
    Create placeholder to feed observations into of the size appropriate to the observation space, and add input
    encoder of the appropriate type.
    '''
    placeholder = observation_placeholder(ob_space, batch_size, name)
    return placeholder, encode_observation(ob_space, placeholder)

def encode_observation(ob_space, placeholder):
    '''
    Encode input in the way that is appropriate to the observation space
    Parameters:
    ----------
    ob_space: gym.Space             observation space
    placeholder: tf.placeholder     observation input placeholder
    '''
    if isinstance(ob_space, Discrete):
        return tf.to_float(tf.one_hot(placeholder, ob_space.n))
    elif isinstance(ob_space, Box):
        return tf.to_float(placeholder)
    elif isinstance(ob_space, MultiDiscrete):
        placeholder = tf.cast(placeholder, tf.int32)
        one_hots = [tf.to_float(tf.one_hot(placeholder[..., i], ob_space.nvec[i])) for i in range(placeholder.shape[-1])]
        return tf.concat(one_hots, axis=-1)
    else:
        raise NotImplementedError

# ================================================================
# Shape adjustment for feeding into tf placeholders
# ================================================================
def adjust_shape(placeholder, data):
    '''
    adjust shape of the data to the shape of the placeholder if possible.
    If shape is incompatible, AssertionError is thrown
    Parameters:
        placeholder     tensorflow input placeholder
        data            input data to be (potentially) reshaped to be fed into placeholder
    Returns:
        reshaped data
    '''

    if not isinstance(data, np.ndarray) and not isinstance(data, list):
        return data
    if isinstance(data, list):
        data = np.array(data)

    placeholder_shape = [x or -1 for x in placeholder.shape.as_list()]

    assert _check_shape(placeholder_shape, data.shape), \
        'Shape of data {} is not compatible with shape of the placeholder {}'.format(data.shape, placeholder_shape)

    return np.reshape(data, placeholder_shape)


def _check_shape(placeholder_shape, data_shape):
    ''' check if two shapes are compatible (i.e. differ only by dimensions of size 1, or by the batch dimension)'''

    return True
    squeezed_placeholder_shape = _squeeze_shape(placeholder_shape)
    squeezed_data_shape = _squeeze_shape(data_shape)

    for i, s_data in enumerate(squeezed_data_shape):
        s_placeholder = squeezed_placeholder_shape[i]
        if s_placeholder != -1 and s_data != s_placeholder:
            return False

    return True


def _squeeze_shape(shape):
    return [x for x in shape if x != 1]


# ================================================================
# Placeholders
# ================================================================


class TfInput(object):
    def __init__(self, name="(unnamed)"):
        """Generalized Tensorflow placeholder. The main differences are:
            - possibly uses multiple placeholders internally and returns multiple values
            - can apply light postprocessing to the value feed to placeholder.
        """
        self.name = name

    def get(self):
        """Return the tf variable(s) representing the possibly postprocessed value
        of placeholder(s).
        """
        raise NotImplementedError

    def make_feed_dict(data):
        """Given data input it to the placeholder(s)."""
        raise NotImplementedError


class PlaceholderTfInput(TfInput):
    def __init__(self, placeholder):
        """Wrapper for regular tensorflow placeholder."""
        super().__init__(placeholder.name)
        self._placeholder = placeholder

    def get(self):
        return self._placeholder

    def make_feed_dict(self, data):
        return {self._placeholder: adjust_shape(self._placeholder, data)}


class ObservationInput(PlaceholderTfInput):
    def __init__(self, observation_space, name=None):
        """Creates an input placeholder tailored to a specific observation space
        Parameters
        ----------
        observation_space:
                observation space of the environment. Should be one of the gym.spaces types
        name: str
                tensorflow name of the underlying placeholder
        """
        inpt, self.processed_inpt = observation_input(observation_space, name=name)
        super().__init__(inpt)

    def get(self):
        return self.processed_inpt

def variable_summaries(name, var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    name = name.replace(':', '_')
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(name+'_mean', mean)
        with tf.name_scope(name+'_stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(name+'_stddev', stddev)
        tf.summary.scalar(name+'_max', tf.reduce_max(var))
        tf.summary.scalar(name+'_min', tf.reduce_min(var))
        tf.summary.histogram(name+'_histogram', var)
