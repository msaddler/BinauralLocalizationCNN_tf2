import os
import sys
import pdb
import numpy as np
import tensorflow as tf

import util_signal


def cochlea(tensor_input,
            sr_input=20e3,
            sr_cochlea=None,
            sr_output=None,
            dtype=tf.float32,
            config_filterbank={},
            config_subband_processing={},
            kwargs_fir_lowpass_filter_input={},
            kwargs_fir_lowpass_filter_output={}):
    """
    """
    # Initialize sequential keras model for cochlear processing stages 
    model = tf.keras.Sequential()
    
    # Resample audio from sr_input to sr_cochlea
    if sr_cochlea is None:
        sr_cochlea = sr_input
    if sr_output is None:
        sr_output = sr_input
    if (not sr_input == sr_cochlea) or (kwargs_fir_lowpass_filter_input):
        lambda_resample = util_signal.tf_fir_resample(
            tensor_input,
            sr_input=sr_input,
            sr_output=sr_cochlea,
            kwargs_fir_lowpass_filter=kwargs_fir_lowpass_filter_input,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(lambda_resample, name='resample_input'))
    
    # Convert audio to subbands as specified by config_filterbank
    filterbank_mode = config_filterbank.pop('mode', None)
    if config_filterbank:
        print('[cochlea] converting audio to subbands using {}'.format(filterbank_mode))
    if filterbank_mode is None:
        assert not config_filterbank, "filterbank_mode must be specified in config_filterbank"
        assert len(tensor_input.shape) >= 3, "shape must be [batch, time, freq, (channel)] to skip filterbank"
    elif filterbank_mode == 'half_cosine_filterbank':
        filterbank_io_function = util_signal.half_cosine_filterbank(
            model(tensor_input),
            sr_cochlea,
            **config_filterbank,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(filterbank_io_function, dtype=dtype, name='filterbank'))
    else:
        raise NotImplementedError("filterbank_mode={} not recognized".format(filterbank_mode))
    
    # Half-wave rectify subbands
    if config_subband_processing.get('rectify', False):
        model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x), dtype=dtype, name='relu_subbands'))
        print('[cochlea] half-wave rectified subbands')
    # Resample subbands from sr_cochlea to sr_output
    if (not sr_output == sr_cochlea) or (kwargs_fir_lowpass_filter_output):
        lambda_resample = util_signal.tf_fir_resample(
            model(tensor_input),
            sr_input=sr_cochlea,
            sr_output=sr_output,
            kwargs_fir_lowpass_filter=kwargs_fir_lowpass_filter_output,
            return_io_function=True)
        model.add(tf.keras.layers.Lambda(lambda_resample, dtype=dtype, name='resample_subbands'))
        print('[cochlea] resampled subbands from {} Hz to {} Hz with filter: {}'.format(
            int(sr_cochlea), int(sr_output), kwargs_fir_lowpass_filter_output))
        # Half-wave rectify subbands again after resampling
        if config_subband_processing.get('rectify', False):
            model.add(tf.keras.layers.Lambda(lambda x: tf.nn.relu(x), dtype=dtype, name='relu_resampled_subbands'))
            print('[cochlea] half-wave rectified resampled subbands')
    
    # Apply power compression
    power_compression = config_subband_processing.get('power_compression', None)
    if power_compression is not None:
        
        @tf.custom_gradient
        def stable_power_compression(x):
            """
            This function wraps around tf.math.pow and adds a custom gradient
            for stable back-propagation. Gradients are clipped to [-1, 1] and
            nan values are removed. Function is modeled after tfcochleagram.
            """
            y = tf.math.pow(x, power_compression)
            def grad(dy):
                g = power_compression * tf.math.pow(x, power_compression - 1)
                nan_mask = tf.math.is_nan(g)
                nan_replace = tf.ones_like(g)
                return dy * tf.where(nan_mask, nan_replace, tf.clip_by_value(g, -1, 1))
            return y, grad
        
        model.add(tf.keras.layers.Lambda(stable_power_compression, dtype=dtype, name='stable_power_compression'))
        print('[cochlea] applied {} power compression to subbands'.format(power_compression))
    
    tensor_output = None
    if tensor_input is not None:
        tensor_output = model(tensor_input)
    return tensor_output, model


def random_slice(tensor_input, slice_length, axis=-1, buffer=None):
    """
    """
    if isinstance(buffer, int):
        buffer_start = buffer
        buffer_end = buffer
    elif isinstance(buffer, tuple):
        buffer_start, buffer_end = buffer
    else:
        buffer_start = 0
        buffer_end = 0
    idx = tf.random.uniform(
        shape=(),
        minval=buffer_start,
        maxval=tensor_input.shape[axis] + 1 - buffer_end - slice_length,
        dtype=tf.int64)
    slice_index = [slice(None, None, None) for _ in range(len(tensor_input.shape))]
    slice_index[axis] = slice(idx, idx + slice_length, None)
    return tensor_input[slice_index]
