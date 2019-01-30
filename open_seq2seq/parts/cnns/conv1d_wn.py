import tensorflow as tf
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn_ops

class Conv1D_WN(tf.layers.Conv1D):
  def __init__(self, filters,
               kernel_size,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer=tf.zeros_initializer(),
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               name=None,
               bias_init=False,
               **kwargs):
    super(Conv1D_WN, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        trainable=trainable,
        name=name, **kwargs)
    self.bias_init = bias_init

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    kernel_shape = self.kernel_size + (input_dim, self.filters)

    direction = self.add_weight(
        name='direction',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        # regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    direction_norm = tf.norm(direction.initialized_value(), axis=[0, 1])
    weight = self.add_weight(
        name='weight',
        shape=[self.filters],
        initializer=lambda shape, dtype, partition_info: direction_norm,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    # Try to see if this avoids numerical instability
    self.kernel = tf.reshape(weight, [1, 1, self.filters]) * tf.nn.l2_normalize(
        # direction, [0, 1], epsilon=1e-6)
         direction, epsilon=1e-6)
    if self.use_bias:
      if self.bias_init:
        bias1 = self.add_weight(
            name='pass_bias',
            shape=(int(self.filters/2),),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
        bias2 = self.add_weight(
            name='gate_bias',
            shape=(int(self.filters/2),),
            initializer=tf.constant_initializer(value=2),
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
        self.bias = tf.concat([bias1, bias2], 0)
      else:
        self.bias = self.add_weight(
            name='bias',
            shape=(self.filters,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)
    else:
      self.bias = None
    self.input_spec = InputSpec(ndim=self.rank + 2,
                                axes={channel_axis: input_dim})
    if self.padding == 'causal':
      op_padding = 'valid'
    else:
      op_padding = self.padding
    self._convolution_op = nn_ops.Convolution(
        input_shape,
        filter_shape=self.kernel.get_shape(),
        dilation_rate=self.dilation_rate,
        strides=self.strides,
        padding=op_padding.upper(),
        data_format=conv_utils.convert_data_format(self.data_format,
                                                   self.rank + 2))
    self.built = True

def conv1d_wn(inputs,
              filters,
              kernel_size,
              strides=1,
              padding='valid',
              data_format='channels_last',
              dilation_rate=1,
              activation=None,
              use_bias=True,
              kernel_initializer=None,
              bias_initializer=tf.zeros_initializer(),
              kernel_regularizer=None,
              bias_regularizer=None,
              activity_regularizer=None,
              kernel_constraint=None,
              bias_constraint=None,
              trainable=True,
              name=None,
              reuse=None,
              bias_init=False):
  layer = Conv1D_WN(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      data_format=data_format,
      dilation_rate=dilation_rate,
      activation=activation,
      use_bias=use_bias,
      kernel_initializer=kernel_initializer,
      bias_initializer=bias_initializer,
      kernel_regularizer=kernel_regularizer,
      bias_regularizer=bias_regularizer,
      activity_regularizer=activity_regularizer,
      kernel_constraint=kernel_constraint,
      bias_constraint=bias_constraint,
      trainable=trainable,
      name=name,
      bias_init=bias_init,
      _reuse=reuse,
      _scope=name)
  return layer.apply(inputs)
