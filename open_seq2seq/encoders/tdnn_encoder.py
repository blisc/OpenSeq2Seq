# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import tensorflow as tf
from tensorflow.python.framework import ops

from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv,\
                                                conv_ln_actv, conv_in_actv,\
                                                conv_bn_res_bn_actv, \
                                                conv2d_actv, conv2d_bn_actv, \
                                                conv1d_wn_actv, conv2d_wn_actv
from open_seq2seq.parts.convs2s.conv_wn_layer import Conv1DNetworkNormalized
from open_seq2seq.parts.convs2s.ffn_wn_layer import FeedFowardNetworkNormalized
from open_seq2seq.parts.convs2s.utils import gated_linear_units
from .encoder import Encoder


class TDNNEncoder(Encoder):
  """General time delay neural network (TDNN) encoder. Fully convolutional model
  """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'convnet_layers': list,
        'activation_fn': None,  # any valid callable
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'normalization': [None, 'batch_norm', 'layer_norm', 'instance_norm', 'weight_norm'],
        'bn_momentum': float,
        'bn_epsilon': float,
    })

  def __init__(self, params, model, name="w2l_encoder", mode='train'):
    """TDNN encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **dropout_keep_prop** (float) --- keep probability for dropout.
    * **convnet_layers** (list) --- list with the description of convolutional
      layers. For example::
        "convnet_layers": [
          {
            "type": "conv1d", "repeat" : 5,
            "kernel_size": [7], "stride": [1],
            "num_channels": 250, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 3,
            "kernel_size": [11], "stride": [1],
            "num_channels": 500, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [32], "stride": [1],
            "num_channels": 1000, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [1], "stride": [1],
            "num_channels": 1000, "padding": "SAME"
          },
        ]
    * **activation_fn** --- activation function to use.
    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **normalization** --- normalization to use. Accepts [None, 'batch_norm'].
      Use None if you don't want to use normalization. Defaults to 'batch_norm'.
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
    """
    super(TDNNEncoder, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Wav2Letter like encoder.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              "source_tensors": [
                src_sequence (shape=[batch_size, sequence length, num features]),
                src_length (shape=[batch_size])
              ]
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
          'src_length': tensor, shape=[batch_size]
        }
    """

    source_sequence, src_length = input_dict['source_tensors']

    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')
    normalization = self.params.get('normalization', 'batch_norm')

    normalization_params = {}
    if normalization is None:
      conv_block = conv_actv
    elif normalization == "batch_norm":
      conv_block = conv_bn_actv
      normalization_params['bn_momentum'] = self.params.get(
          'bn_momentum', 0.90)
      normalization_params['bn_epsilon'] = self.params.get('bn_epsilon', 1e-3)
    elif normalization == "layer_norm":
      conv_block = conv_ln_actv
    elif normalization == "instance_norm":
      conv_block = conv_in_actv
    elif normalization == "weight_norm":
      conv_block = conv1d_wn_actv
    else:
      raise ValueError("Incorrect normalization")

    conv_inputs = source_sequence
    if data_format == 'channels_last':
      conv_feats = conv_inputs  # B T F
    else:
      conv_feats = tf.transpose(conv_inputs, [0, 2, 1])  # B F T

    residual_aggregation = []

    # ----- Convolutional layers ---------------------------------------------
    convnet_layers = self.params['convnet_layers']

    for idx_convnet in range(len(convnet_layers)):
      layer_type = convnet_layers[idx_convnet]['type']
      layer_repeat = convnet_layers[idx_convnet]['repeat']
      ch_out = convnet_layers[idx_convnet]['num_channels']
      if self.params["activation_fn"] is gated_linear_units:
        ch_out = ch_out * 2
      kernel_size = convnet_layers[idx_convnet]['kernel_size']
      strides = convnet_layers[idx_convnet]['stride']
      padding = convnet_layers[idx_convnet]['padding']
      dilation = convnet_layers[idx_convnet]['dilation']
      dropout_keep = convnet_layers[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0
      residual = convnet_layers[idx_convnet].get('residual', False)
      residual_dense = convnet_layers[idx_convnet].get('residual_dense', False)
      normalization_params['dropout'] = dropout_keep

      if residual:
        layer_res = conv_feats
        if residual_dense:
          residual_aggregation.append(layer_res)
          layer_res = residual_aggregation
      for idx_layer in range(layer_repeat):
        if padding == "VALID":
          src_length = (src_length - kernel_size[0]) // strides[0] + 1
        else:
          src_length = (src_length + strides[0] - 1) // strides[0]
        if residual and idx_layer == layer_repeat - 1:
          conv_feats = conv_bn_res_bn_actv(
              layer_type=layer_type,
              name="conv{}{}".format(
                  idx_convnet + 1, idx_layer + 1),
              inputs=conv_feats,
              res_inputs=layer_res,
              filters=ch_out,
              kernel_size=kernel_size,
              activation_fn=self.params['activation_fn'],
              strides=strides,
              padding=padding,
              dilation=dilation,
              regularizer=regularizer,
              training=training,
              data_format=data_format,
              **normalization_params
          )
        else:
          conv_feats = conv_block(
              layer_type=layer_type,
              name="conv{}{}".format(
                  idx_convnet + 1, idx_layer + 1),
              inputs=conv_feats,
              filters=ch_out,
              kernel_size=kernel_size,
              activation_fn=self.params['activation_fn'],
              strides=strides,
              padding=padding,
              dilation=dilation,
              regularizer=regularizer,
              training=training,
              data_format=data_format,
              **normalization_params
          )
        conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

    outputs = conv_feats

    if data_format == 'channels_first':
      outputs = tf.transpose(outputs, [0, 2, 1])

    # if self._mode == "train":
    #   for var in tf.trainable_variables():
    #     if "kernel" in var.name:
    #       if var.dtype.base_dtype == tf.float16:
    #         tf.add_to_collection(
    #             'REGULARIZATION_FUNCTIONS', (var, regularizer)
    #         )
    #       else:
    #         print("Added regularizer to {}".format(var.name))
    #         tf.add_to_collection(
    #             ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(var)
    #         )

    return {
        'outputs': outputs,
        'src_length': src_length,
    }

class TDNNEncoder2(Encoder):
  """General time delay neural network (TDNN) encoder. Fully convolutional model
  """

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'convnet_layers': list,
        'activation_fn': None,  # any valid callable
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'normalization': [None, 'batch_norm', 'layer_norm', 'weight_norm']
    })

  def __init__(self, params, model, name="w2l_encoder", mode='train'):
    """TDNN encoder constructor.

    See parent class for arguments description.

    Config parameters:

    * **dropout_keep_prop** (float) --- keep probability for dropout.
    * **convnet_layers** (list) --- list with the description of convolutional
      layers. For example::
        "convnet_layers": [
          {
            "type": "conv1d", "repeat" : 5,
            "kernel_size": [7], "stride": [1],
            "num_channels": 250, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 3,
            "kernel_size": [11], "stride": [1],
            "num_channels": 500, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [32], "stride": [1],
            "num_channels": 1000, "padding": "SAME"
          },
          {
            "type": "conv1d", "repeat" : 1,
            "kernel_size": [1], "stride": [1],
            "num_channels": 1000, "padding": "SAME"
          },
        ]
    * **activation_fn** --- activation function to use.
    * **data_format** (string) --- could be either "channels_first" or
      "channels_last". Defaults to "channels_last".
    * **normalization** --- normalization to use. Accepts [None, 'batch_norm'].
      Use None if you don't want to use normalization. Defaults to 'batch_norm'.
    * **bn_momentum** (float) --- momentum for batch norm. Defaults to 0.90.
    * **bn_epsilon** (float) --- epsilon for batch norm. Defaults to 1e-3.
    """
    super(TDNNEncoder2, self).__init__(params, model, name, mode)

  def _encode(self, input_dict):
    """Creates TensorFlow graph for Wav2Letter like encoder.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              "source_tensors": [
                src_sequence (shape=[batch_size, sequence length, num features]),
                src_length (shape=[batch_size])
              ]
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'outputs': hidden state, shape=[batch_size, sequence length, n_hidden]
          'src_length': tensor, shape=[batch_size]
        }
    """

    source_sequence, src_length = input_dict['source_tensors']

    training = (self._mode == "train")
    dropout_keep_prob = self.params['dropout_keep_prob'] if training else 1.0
    regularizer = self.params.get('regularizer', None)
    data_format = self.params.get('data_format', 'channels_last')
    normalization = self.params.get('normalization', 'batch_norm')

    if normalization is None:
      conv_block = conv2d_actv
    elif normalization == "batch_norm":
      conv_block = conv2d_bn_actv
    elif normalization == "weight_norm":
      conv_block = conv2d_wn_actv
    else:
      raise ValueError("Incorrect normalization")

    conv_inputs = source_sequence
    if data_format == 'channels_last':
      conv_feats = conv_inputs  # B T F
    else:
      conv_feats = tf.transpose(conv_inputs, [0, 2, 1])  # B F T

    residual_aggregation = []
    conv_feats = tf.expand_dims(conv_feats, axis=1)

    # ----- Convolutional layers ---------------------------------------------
    convnet_layers = self.params['convnet_layers']

    for idx_convnet in range(len(convnet_layers)):
      # layer_type = convnet_layers[idx_convnet]['type']
      layer_repeat = convnet_layers[idx_convnet]['repeat']
      ch_out = convnet_layers[idx_convnet]['num_channels']
      if self.params["activation_fn"] is gated_linear_units:
        ch_out = ch_out * 2
      kernel_size = convnet_layers[idx_convnet]['kernel_size']
      strides = convnet_layers[idx_convnet]['stride']
      padding = convnet_layers[idx_convnet]['padding']
      dilation = convnet_layers[idx_convnet]['dilation']
      dropout_keep = convnet_layers[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0
      residual = convnet_layers[idx_convnet].get('residual', False)
      residual_dense = convnet_layers[idx_convnet].get('residual_dense', False)

      if residual:
        layer_res = [conv_feats]
        if residual_dense:
          residual_aggregation.append(layer_res)
          layer_res = residual_aggregation
      for idx_layer in range(layer_repeat):
        if padding == "VALID":
          src_length = (src_length - kernel_size[0]) // strides[0] + 1
        else:
          src_length = (src_length + strides[0] - 1) // strides[0]

        # Build conv layer
        # layer = Conv1DNetworkNormalized(
        #     in_dim=conv_feats.get_shape().as_list()[-1],
        #     out_dim=ch_out,
        #     kernel_width=kernel_size[0],
        #     mode=self._mode,
        #     layer_id="{}_{}".format(idx_convnet + 1, idx_layer + 1),
        #     hidden_dropout=dropout_keep,
        #     conv_padding=padding,
        #     decode_padding=False,
        #     activation=None,
        #     normalization_type=normalization,
        #     regularizer=regularizer,
        #     dilation=dilation[0],
        #     stride=strides[0]
        # )
        # conv_feats = layer(conv_feats)

        # conv_feats = tf.layers.conv1d(
        #     conv_feats,
        #     # in_dim = conv_feats.get_shape().as_list()[-1],
        #     filters = ch_out,
        #     kernel_size = kernel_size[0],
        #     # mode = self._mode,
        #     name = "{}_{}".format(idx_convnet + 1, idx_layer + 1),
        #     # hidden_dropout = 1.,
        #     padding = padding,
        #     # decode_padding = False,
        #     activation = None,
        #     # normalization_type = normalization,
        #     # regularizer = regularizer,
        #     dilation_rate = dilation[0],
        #     strides = strides[0]
        # )
        # conv_feats = layer(conv_feats)


        # with tf.variable_scope("conv_{}_{}".format(idx_convnet + 1, idx_layer + 1)):
        #   w = tf.get_variable(
        #       'W',
        #       shape=[kernel_size[0], conv_feats.get_shape().as_list()[-1], ch_out],
        #       regularizer=regularizer)

        #   conv_feats = tf.nn.conv1d(
        #   value=conv_feats, filters=w, stride=strides[0], padding=padding)

        conv_feats = conv_block(
            name="conv{}{}".format(
                idx_convnet + 1, idx_layer + 1),
            inputs=conv_feats,
            filters=ch_out,
            kernel_size=kernel_size,
            activation_fn=self.params['activation_fn'],
            strides=strides,
            padding=padding,
            regularizer=regularizer,
            dilation=dilation,
            training=training,
            data_format=data_format,
        )

        # Build residual layers as needed
        # if residual and idx_layer == layer_repeat - 1:
        #   total_res = 0
        #   for i, res in enumerate(layer_res):
        #     res_layer = FeedFowardNetworkNormalized(
        #         in_dim=res.get_shape().as_list()[-1],
        #         out_dim=ch_out,
        #         dropout=1.,
        #         var_scope_name="res_{}".format(idx_convnet + 1),
        #         mode=self._mode,
        #         normalization_type=normalization,
        #         regularizer=regularizer
        #     )
        #     total_res += res_layer(res)
        #   conv_feats += total_res

        # conv_feats = self.params["activation_fn"](conv_feats)
        conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

    outputs = conv_feats

    if data_format == 'channels_first':
      outputs = tf.transpose(outputs, [0, 2, 1])

    if self._mode == "train":
      for var in tf.trainable_variables():
        if "kernel" in var.name or "weight" in var.name:
          # if var.dtype.base_dtype == tf.float16:
          #   tf.add_to_collection(
          #       'REGULARIZATION_FUNCTIONS', (var, regularizer)
          #   )
          # else:
          print("Added regularizer to {}".format(var.name))
          tf.add_to_collection(
              ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(var)
          )
    outputs = tf.squeeze(outputs, axis=1)

    return {
        'outputs': outputs,
        'src_length': src_length,
    }
