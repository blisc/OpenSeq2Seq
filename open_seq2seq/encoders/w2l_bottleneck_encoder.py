# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

from open_seq2seq.parts.cnns.conv_blocks import conv_actv, conv_bn_actv, conv_ln_actv, conv_in_actv, conv_bn_res_bn_actv
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.framework import ops

import tensorflow as tf
from .encoder import Encoder

class Wave2LetterBottleNeckEncoder(Encoder):
  """Wave2Letter like encoder. Fully convolutional model"""

  @staticmethod
  def get_required_params():
    return dict(Encoder.get_required_params(), **{
        'dropout_keep_prob': float,
        'convnet_blocks': list,
        'activation_fn': None,  # any valid callable
    })

  @staticmethod
  def get_optional_params():
    return dict(Encoder.get_optional_params(), **{
        'data_format': ['channels_first', 'channels_last'],
        'normalization': [None, 'batch_norm', 'layer_norm', 'instance_norm'],
        'bn_momentum': float,
        'bn_epsilon': float,
        'enable_rnn': bool,
        'rnn_cell_size': int,
        'rnn_layers': int
    })

  def __init__(self, params, model, name="w2l_encoder", mode='train'):
    """Wave2Letter like encoder constructor.

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
    super(Wave2LetterBottleNeckEncoder, self).__init__(params, model, name, mode)

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
    else:
      raise ValueError("Incorrect normalization")

    conv_inputs = source_sequence
    if data_format == 'channels_last':
      conv_feats = conv_inputs  # B T F
    else:
      conv_feats = tf.transpose(conv_inputs, [0, 2, 1])  # B F T

    residual_aggregation = []

    # ----- Convolutional layers ---------------------------------------------
    convnet_blocks = self.params['convnet_blocks']

    for idx_convnet in range(len(convnet_blocks)):
      layer_type = convnet_blocks[idx_convnet]['type']
      block_repeat = convnet_blocks[idx_convnet]['repeat']
      ch_out = convnet_blocks[idx_convnet]['num_channels']
      kernel_size = convnet_blocks[idx_convnet]['kernel_size']
      strides = convnet_blocks[idx_convnet]['stride']
      padding = convnet_blocks[idx_convnet]['padding']
      dilation = convnet_blocks[idx_convnet]['dilation']
      dropout_keep = convnet_blocks[idx_convnet].get(
          'dropout_keep_prob', dropout_keep_prob) if training else 1.0
      residual = convnet_blocks[idx_convnet].get('residual', False)
      residual_dense = convnet_blocks[idx_convnet].get('residual_dense', False)

      if residual:
        layer_res = conv_feats
        if residual_dense:
          residual_aggregation.append(layer_res)
          layer_res = residual_aggregation
      # not a block
      if block_repeat == -1:
        if padding == "VALID":
          src_length = (src_length - kernel_size_layer[0]) // strides[0] + 1
        else:
          src_length = (src_length + strides[0] - 1) // strides[0]
        conv_feats = conv_block(
            layer_type=layer_type,
            name="conv{}0".format(
                idx_convnet + 1),
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
            **normalization_params)
        conv_feats = tf.nn.dropout(x=conv_feats, keep_prob=dropout_keep)
      else:
        for idx_block in range(block_repeat):
          for idx_layer in range(3):
            if padding == "VALID":
              src_length = (src_length - kernel_size_layer[0]) // strides[0] + 1
            else:
              src_length = (src_length + strides[0] - 1) // strides[0]
            if idx_layer < 2:
              ch_out_layer = ch_out / 4
            else:
              ch_out_layer = ch_out
            if idx_layer == 1:
              kernel_size_layer = kernel_size
            else:
              kernel_size_layer = 1
            if residual and idx_layer == 2:
              conv_feats = conv_bn_res_bn_actv(
                  layer_type=layer_type,
                  name="conv{}{}{}".format(
                      idx_convnet + 1, idx_block+1, idx_layer + 1),
                  inputs=conv_feats,
                  res_inputs=layer_res,
                  filters=ch_out_layer,
                  kernel_size=kernel_size_layer,
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
                  name="conv{}{}{}".format(
                      idx_convnet + 1, idx_block+1, idx_layer + 1),
                  inputs=conv_feats,
                  filters=ch_out_layer,
                  kernel_size=kernel_size_layer,
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

    if self.params.get("enable_rnn", False):
      rnn_input = outputs
      rnn_vars = []
      cell_size = self.params.get("rnn_cell_size", 256)
      num_layers = self.params.get("rnn_layers", 1)
      if self._mode == "infer":
        cell = lambda: tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
            cell_size
        )
        cells_fw = [cell() for _ in range(num_layers)]
        cells_bw = [cell() for _ in range(num_layers)]
        (outputs, _, _) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw, cells_bw, rnn_input,
            sequence_length=src_length,
            dtype=rnn_input.dtype,
            time_major=False)
      else:
        rnn_input = tf.transpose(rnn_input, [1, 0, 2])

        rnn_block = tf.contrib.cudnn_rnn.CudnnLSTM(
            num_layers=num_layers,
            num_units=cell_size,
            direction=cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION,
            dtype=rnn_input.dtype,
            name="cudnn_rnn"
        )
        rnn_block.build(rnn_input.get_shape())
        outputs, _ = rnn_block(rnn_input)
        outputs = tf.transpose(outputs, [1, 0, 2])
        rnn_vars += rnn_block.trainable_variables

      if regularizer and training:
        for weights in rnn_vars:
          if "bias" not in weights.name:
            # print("Added regularizer to {}".format(weights.name))
            if weights.dtype.base_dtype == tf.float16:
              tf.add_to_collection(
                  'REGULARIZATION_FUNCTIONS', (weights, regularizer)
              )
            else:
              tf.add_to_collection(
                  ops.GraphKeys.REGULARIZATION_LOSSES, regularizer(weights)
              )

    return {
        'outputs': outputs,
        'src_length': src_length,
    }
