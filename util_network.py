import os
import sys
import pdb
import copy
import numpy as np
import tensorflow as tf

import util_signal


def build_network(tensor_input, list_layer_dict, n_classes_dict={}):
    """
    Build tensorflow graph for a feedforward neural network given an
    input tensor and list of layer descriptions
    """
    tensor_output = tensor_input
    tensors_dict = {}
    for layer_dict in list_layer_dict:
        layer_type = layer_dict['layer_type'].lower()
        # Select a layer constructor
        if ('batchnormalization' in layer_type) or ('batch_normalization' in layer_type):
            layer = tf.keras.layers.BatchNormalization
        elif ('layernormalization' in layer_type) or ('layer_normalization' in layer_type):
            layer = tf.keras.layers.LayerNormalization
        elif ('groupnormalization' in layer_type) or ('group_normalization' in layer_type):
            layer = tf.keras.layers.GroupNormalization
        elif 'conv2d' in layer_type:
            layer = PaddedConv2D
            assert len(tensor_output.shape) == 4
        elif 'dense' in layer_type:
            layer = tf.keras.layers.Dense
        elif 'dropout' in layer_type:
            layer = tf.keras.layers.Dropout
        elif 'maxpool' in layer_type:
            layer = tf.keras.layers.MaxPool2D
        elif 'reshape' in layer_type:
            layer = tf.keras.layers.Reshape
        elif 'flatten' in layer_type:
            layer = tf.keras.layers.Flatten
        elif 'relu' in layer_type:
            layer = tf.keras.layers.ReLU
        elif 'fc_top' in layer_type:
            layer = lambda **args: DenseTaskHeads(n_classes_dict=n_classes_dict, **args)
        else:
            msg = "layer_type={} not recognized".format(layer_type)
            raise NotImplementedError(msg)
        # Build the layer and store named tensors in `tensors_dict`
        args = copy.deepcopy(layer_dict.get('args', {}))
        tensor_output = layer(**args)(tensor_output)
        if args.get('name', None) is not None:
            tensors_dict[args['name']] = tensor_output
    return tensor_output, tensors_dict


def same_pad_along_axis(tensor_input,
                        kernel_length=1,
                        stride_length=1,
                        axis=1,
                        **kwargs):
    """
    Adds 'SAME' padding to only specified axis of tensor_input
    for 2D convolution
    """
    x = tensor_input.shape.as_list()[axis]
    if x % stride_length == 0:
        p = kernel_length - stride_length
    else:
        p = kernel_length - (x % stride_length)
    p = tf.math.maximum(p, 0)
    paddings = [(0, 0)] * len(tensor_input.shape)
    paddings[axis] = (p // 2, p - p // 2)
    return tf.pad(tensor_input, paddings, **kwargs)


def PaddedConv2D(filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='VALID',
                 **kwargs):
    """
    Wrapper function around tf.keras.layers.Conv2D to support
    custom padding options
    """
    if padding.upper() == 'VALID_TIME':
        pad_function = lambda x : same_pad_along_axis(
            x,
            kernel_length=kernel_size[0],
            stride_length=strides[0],
            axis=1)
        padding = 'VALID'
    else:
        pad_function = lambda x: x
    def layer(tensor_input):
        conv2d_layer = tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            **kwargs)
        return conv2d_layer(pad_function(tensor_input))
    return layer


def DenseTaskHeads(n_classes_dict={}, name='logits', **kwargs):
    """
    Dense layer for each task head specified in n_classes_dict
    """
    def layer(tensor_input):
        tensors_logits = {}
        for key in sorted(n_classes_dict.keys()):
            if len(n_classes_dict.keys()) > 1:
                classification_name = '{}_{}'.format(name, key)
            else:
                classification_name = name
            classification_layer = tf.keras.layers.Dense(
                units=n_classes_dict[key],
                name=classification_name,
                **kwargs)
            tensors_logits[key] = classification_layer(tensor_input)
        return tensors_logits
    return layer
