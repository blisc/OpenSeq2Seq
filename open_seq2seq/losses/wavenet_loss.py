# Copyright (c) 2018 NVIDIA Corporation

import tensorflow as tf

from .loss import Loss

class WavenetLoss(Loss):

	def __init__(self, params, model, name="wavenet_loss"):
		super(WavenetLoss, self).__init__(params, model, name)
		self._n_feats = self._model.get_data_layer().params["num_audio_features"]

	def get_required_params(self):
		return dict(Loss.get_required_params(), **{
			"quantization_channels": int,
			"batch_size": int
		})

	def get_optional_params(self):
		return {}

	def _compute_loss(self, input_dict):
		channels = self.params["quantization_channels"]
		batch_size = self.params["batch_size"]

		logits = input_dict["decoder_output"]["logits"]
		outputs = tf.squeeze(input_dict["decoder_output"]["outputs"][0], 2)

		print(logits.get_shape(), outputs.get_shape())

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
		loss = reduced_loss = tf.reduce_mean(loss)

		tf.summary.scalar("loss", loss)

		return loss
		