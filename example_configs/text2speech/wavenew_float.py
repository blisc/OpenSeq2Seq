# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Text2SpeechWavenet
from open_seq2seq.encoders import WavenewEncoder
from open_seq2seq.decoders import FakeDecoder
from open_seq2seq.losses import WavenetLoss
from open_seq2seq.data import WavenetDataLayer
from open_seq2seq.optimizers.lr_policies import exp_decay

base_model = Text2SpeechWavenet

base_params = {
  "random_seed": 0,
  "use_horovod": True,
  "max_steps": 50000,

  "num_gpus": 1,
  "batch_size_per_gpu": 2,

  # [TODO] add logging params
  "save_summaries_steps": 100,
  "print_loss_steps": 100,
  "print_samples_steps": 1000,
  "eval_steps": 1000,
  "save_checkpoint_steps": 5000,
  "logdir": "result/wavenet-LJ-float",

  "optimizer": "Adam",
  "optimizer_params": {},
  "lr_policy": exp_decay,
  "lr_policy_params": {
    "learning_rate": 1e-3,
    "decay_steps": 20000,
    "decay_rate": 0.1,
    "use_staircase_decay": False,
    "begin_decay_at": 15000,
    "min_lr": 1e-5,
  },
  "dtype": tf.float32,
  "regularizer": tf.contrib.layers.l2_regularizer,
  "regularizer_params": {
    "scale": 1e-6
  },
  "initializer": tf.contrib.layers.xavier_initializer,

  "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                'variable_norm', 'gradient_norm', 'global_gradient_norm'],

  "encoder": WavenewEncoder,
  "encoder_params": {
    "activation_fn": tf.nn.relu,
    "dropout_keep_prob": 0.7,
    "quantization_channels": 256,
    "convnet_layers": [
      {
        "kernel_size": [1,8], "stride": [1,4],
        "num_channels": 32, "dropout_keep_prob": 0.9,
      },
      # {
      #   "kernel_size": [1,8], "stride": [1,2],
      #   "num_channels": 256, "dropout_keep_prob": 0.8,
      # },
      {
        "kernel_size": [1,8], "stride": [1,4],
        "num_channels": 32, "dropout_keep_prob": 0.8,
      },
      # {
      #   "kernel_size": [1,8], "stride": [1,2],
      #   "num_channels": 512, "dropout_keep_prob": 0.8,
      # },
      {
        "kernel_size": [1,8], "stride": [1,4],
        "num_channels": 64,"dropout_keep_prob": 0.7,
      },
      # {
      #   "kernel_size": [1,8], "stride": [1,2],
      #   "num_channels": 768, "dropout_keep_prob": 0.7,
      # },
      {
        "kernel_size": [1,8], "stride": [1,4],
        "num_channels": 64, "dropout_keep_prob": 0.6,
      },
      # {
      #   "kernel_size": [1,8], "stride": [1,2],
      #   "num_channels": 1024, "dropout_keep_prob": 0.6,
      # }
    ],

  },

  "decoder": FakeDecoder,

  "loss": WavenetLoss,
  "loss_params": {
    "quantization_channels": 256,
  },

  "data_layer": WavenetDataLayer,
  "data_layer_params": {
    "dataset": "LJ",
    "num_audio_features": 513,
    "dataset_location": "/data/speech/LJSpeech/wavs/"
  }
}

train_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/speech/LJSpeech/train_32.csv",
    ],
    "shuffle": True,
  },
}

eval_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/speech/LJSpeech/val_32.csv",
    ],
    "shuffle": False,
  },
}

infer_params = {
  "data_layer_params": {
    "dataset_files": [
      "/data/speech/LJSpeech/test.csv",
    ],
    "shuffle": False,
  },
}

interactive_infer_params = {
  "data_layer_params": {
    "dataset_files": [],
    "shuffle": False,
  },
}
