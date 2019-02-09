import tensorflow as tf
import numpy as np

# ------Helper Functions for creating neural networks ---------
def create_conv_2dlayer(input,
                        num_input_channels,
                        filter_size,
                        num_output_channel,
                        stride=1,
                        relu=True,
                        pooling=False,
                        padding='valid',
                        data_format='channels_last'):  # Number of filters.

    layer = tf.layers.conv2d(inputs=input, filters=num_output_channel,
                             kernel_size=[filter_size, filter_size],
                             padding=padding,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.05),
                             bias_initializer=tf.constant_initializer(0.05),
                             data_format= data_format)

    if pooling:
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 3, 3, 1],
                               strides=[1, 1, 1, 1],
                               padding='VALID')

    if relu:
        layer = tf.nn.relu(layer)

    return layer


def fully_connected_layer(input,
                          num_inputs,
                          num_outputs,
                          activation=None):

    weights = tf.get_variable('weights', shape=[num_inputs, num_outputs])
    biases = tf.get_variable('biases', shape=num_outputs)

    layer = tf.matmul(input, weights) + biases

    if activation is not None:
        if activation == 'relu':
            layer = tf.nn.relu(layer)

        elif activation == 'softmax':
            layer = tf.nn.softmax(layer)

    return layer


def flatten_layer(layer):
    layer_shape = layer.get_shape()  # layer = [num_images, img_height, img_width, num_channels]
    num_features = layer_shape[1:].num_elements()  # Total number of elements in the network
    layer_flat = tf.reshape(layer, [-1, num_features])  # -1 means total size of dimension is unchanged

    return layer_flat, num_features


def create_conv_3dlayer(input,
                        filter_width,
                        filter_height,
                        filter_depth,
                        stride=1,
                        num_input_channels=1,
                        num_output_channels=1,
                        relu=True):

    layer_3d = input

    shape = [filter_depth, filter_width, filter_height, num_input_channels, num_output_channels]
    weights = tf.get_variable(name='weights', shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.05))
    biases = tf.get_variable(name='biases', shape=[num_output_channels], initializer=tf.constant_initializer(0.05))

    layer = tf.nn.conv3d(input=layer_3d, filter=weights, strides=[1, stride, 1, 1, 1], padding='VALID') #, data_format='NDHWC'
    layer += biases

    if relu:
        layer = tf.nn.relu(layer)

    return layer



########################################################
# ---Configuration for different parallel models-------#
# ---Contain BASS-NET, 3D Conv layers -----------------#
########################################################
# --------Parameter Sharing ----------


