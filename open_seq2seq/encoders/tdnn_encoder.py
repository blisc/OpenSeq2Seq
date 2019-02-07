# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import math
import tensorflow as tf

from .encoder import Encoder
from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv,\
                                                conv_ln_actv, conv_in_actv,\
                                                conv_bn_res_bn_actv,\
                                                conv1d_wn_actv_res, conv_res_bn_actv, conv_res_ln_actv,\
                                                conv_res_actv
from open_seq2seq.parts.convs2s.ffn_wn_layer import FeedFowardNetworkNormalized
from open_seq2seq.parts.convs2s.utils import gated_unit


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
        # 'res_before_actv': bool,
        'wn_bias_init': bool,
        'gate_activation_fn': None,
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
    res_normalization = normalization = self.params.get('normalization', 'batch_norm')

    normalization_params = {}
    # if normalization is None:
    #   conv_block = conv_actv
    if normalization == "batch_norm":
      conv_block = conv_res_bn_actv
      normalization_params['bn_momentum'] = self.params.get(
          'bn_momentum', 0.90)
      normalization_params['bn_epsilon'] = self.params.get('bn_epsilon', 1e-3)
      normalization_params['training'] = training
      res_factor = 1
      res_normalization = None
    elif normalization == "layer_norm":
      conv_block = conv_res_ln_actv
      res_factor = 1
      res_normalization = None
    # elif normalization == "instance_norm":
    #   conv_block = conv_in_actv
    elif normalization == "weight_norm":
      conv_block = conv1d_wn_actv_res
      res_factor = 0.5
      normalization_params["bias_init"] = self.params.get("wn_bias_init", False)
    elif normalization == None:
      conv_block = conv_res_actv
      res_factor = 1
    else:
      raise ValueError("Incorrect normalization")

    using_gated_unit = False
    if self.params["activation_fn"] is gated_unit:
      gate_activation_fn = self.params.get("gate_activation_fn", None)
      self.params["activation_fn"] = lambda x: gated_unit(x, gate_activation_fn)
      using_gated_unit = True

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
      ch_out_c = ch_out_r = convnet_layers[idx_convnet]['num_channels']
      if using_gated_unit:
        ch_out_c = int(ch_out_c * math.sqrt(2))
        ch_out_c += ch_out_c % 2
        ch_out_r = int(ch_out_c * res_factor)
      kernel_size = convnet_layers[idx_convnet]['kernel_size']
      strides = convnet_layers[idx_convnet]['stride']
      padding = convnet_layers[idx_convnet]['padding']
      dilation = convnet_layers[idx_convnet]['dilation']
      dropout_keep = convnet_layers[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0
      residual = convnet_layers[idx_convnet].get('residual', False)
      final_skip = convnet_layers[idx_convnet].get('final_skip', False)

      # If residual is "res", "dense", or "skip"
      if residual:
        # Normal "res" - only skips one convolution block
        layer_res = [conv_feats]
        # For "dense" or "skip", we want to aggregate residual connections
        if residual != "res":
          residual_aggregation.append(layer_res[0])
          # For "dense", we want to pass every residual to current block
          if residual != "skip":
            layer_res = residual_aggregation
      for idx_layer in range(layer_repeat):
        if padding == "VALID":
          src_length = (src_length - kernel_size[0]) // strides[0] + 1
        else:
          src_length = (src_length + strides[0] - 1) // strides[0]
        total_res = None
        scale = 1
        if residual == "skip" and final_skip and idx_layer == layer_repeat - 1:
          total_res = 0
          for i, res in enumerate(residual_aggregation):
            res_layer = FeedFowardNetworkNormalized(
                in_dim=res.get_shape().as_list()[-1],
                out_dim=ch_out_r,
                dropout=1.,
                var_scope_name="conv{}{}/res_{}".format(idx_convnet + 1, idx_layer + 1, i + 1),
                mode=self._mode,
                normalization_type=res_normalization,
                regularizer=regularizer
            )
            total_res += res_layer(res)
        elif residual and idx_layer == layer_repeat - 1:
          scale += 1
          total_res = 0
          for i, res in enumerate(layer_res):
            res_layer = FeedFowardNetworkNormalized(
                in_dim=res.get_shape().as_list()[-1],
                out_dim=ch_out_r,
                dropout=1.,
                var_scope_name="conv{}{}/res_{}".format(idx_convnet + 1, idx_layer + 1, i + 1),
                mode=self._mode,
                normalization_type=res_normalization,
                regularizer=regularizer
            )
            total_res += res_layer(res)

        scale = math.sqrt(1/float(scale))

        conv_feats = conv_block(
            layer_type=layer_type,
            name="conv{}{}".format(
                idx_convnet + 1, idx_layer + 1),
            inputs=conv_feats,
            res=total_res,
            filters=ch_out_c,
            kernel_size=kernel_size,
            activation_fn=self.params["activation_fn"],
            strides=strides,
            padding=padding,
            dilation=dilation,
            regularizer=regularizer,
            data_format=data_format,
            **normalization_params
        )

        if normalization == "weight_norm":
          conv_feats *= scale
        conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)

    outputs = conv_feats
    # if normalization == "weight_norm":
    #   # Reuse dropout probability from last layer
    #   outputs = tf.nn.dropout(x=outputs, keep_prob=dropout_keep)

    if data_format == 'channels_first':
      outputs = tf.transpose(outputs, [0, 2, 1])

    return {
        'outputs': outputs,
        'src_length': src_length,
    }
