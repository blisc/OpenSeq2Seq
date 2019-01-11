# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders import TDNNEncoder
from open_seq2seq.decoders import JointCTCAttentionDecoder
from open_seq2seq.decoders.rnn_decoders import RNNDecoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import MultiTaskCTCEntropyLoss
from open_seq2seq.optimizers.lr_policies import poly_decay

residual = True
residual_dense = True
repeat_1 = 3
repeat_2 = 3
dropout_factor = 1.
training_set = "libri"
data_aug_enable = False

if training_set == "libri":
    dataset_files = [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv"]
elif training_set == "combined":
    dataset_files = [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv",
            "/data/speech/LibriSpeech/LibriSpeech/data_syn.txt"]
elif training_set == "MAILABS_LibriSpeech":
    dataset_files = [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv",
            "/mnt/hdd/data/MAILABS/train.csv"]
elif training_set == "syn":
    dataset_files = ["/data/speech/LibriSpeech/LibriSpeech/data_syn.txt"]
elif training_set == "combined_33_66":
    dataset_files = [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv",
            "/data/speech/LibriSpeech/LibriSpeech/data_syn.txt",
            "/data/speech/LibriSpeech/LibriSpeech/data_syn.txt"]

data_aug = {}
if data_aug_enable == True:
    data_aug = {
            'time_stretch_ratio': 0.05,
            'noise_level_min': -90,
            'noise_level_max': -60}

base_model = Speech2Text

base_params = {
    "random_seed": 0,
    "use_horovod": True,
    "num_epochs": 200,

    "num_gpus": 8,
    "batch_size_per_gpu": 128,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 2200,
    "eval_steps": 2200,
    "save_checkpoint_steps": 1100,
    "logdir": "nsr_jcr_log",
    # "load_model": "w2l_log_folder",
    # "freeze_variables_regex": "w2l_encoder",

    "optimizer": "Momentum",
    "optimizer_params": {
        "momentum": 0.90,
    },
    "lr_policy": poly_decay,
    "lr_policy_params": {
        "learning_rate": 0.01,
        "min_lr": 1e-5,
        "power": 2.0,
    },
    "larc_params": {
        "larc_eta": 0.001,
    },

    "regularizer": tf.contrib.layers.l2_regularizer,
    "regularizer_params": {
        'scale': 0.001
    },

    "dtype": "mixed",
    "loss_scaling": "Backoff",

    "summaries": ['learning_rate', 'variables', 'gradients', 'larc_summaries',
                  'variable_norm', 'gradient_norm', 'global_gradient_norm'],

    "encoder": TDNNEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [11], "stride": [1],
                "num_channels": 256, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [13], "stride": [1],
                "num_channels": 384, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [17], "stride": [1],
                "num_channels": 512, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [21], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [25], "stride": [1],
                "num_channels": 768, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
                "residual": residual, "residual_dense": residual_dense
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 896, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 0.6 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 1024, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.6 * dropout_factor,
            }
        ],

        "dropout_keep_prob": 0.7 * dropout_factor,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": "batch_norm",
        "activation_fn": lambda x: tf.minimum(tf.nn.relu(x), 20.0),
        "data_format": "channels_last",
    },

    "decoder": JointCTCAttentionDecoder,
    "decoder_params": {

        "attn_decoder": RNNDecoder,
        "attn_decoder_params": {
            'tgt_vocab_size': int,
            'num_rnn_layers': 3,
            'rnn_cell_dim': 256,
            "rnn_unidirectional": False,
            "use_cudnn_rnn": True,
            "rnn_type": tf.contrib.cudnn_rnn.CudnnLSTM,
        },

        "ctc_decoder": FullyConnectedCTCDecoder,
        "ctc_decoder_params": {
            "initializer": tf.contrib.layers.xavier_initializer,
            "use_language_model": False,
        },

        # "beam_search_params": {
        #     "beam_width": 4,
        # },

        # "language_model_params": {
        #     # params for decoding the sequence with language model
        #     "use_language_model": False,
        # },

    },

    "loss": MultiTaskCTCEntropyLoss,
    "loss_params": {

        "seq_loss_params": {
            "offset_target_by_one": False,
            "average_across_timestep": True,
            "do_mask": True,
            "reduce_tgt_size_by_one": True
        },

        "ctc_loss_params": {
        },

        "lambda_value": 0.25,
    }
    # "loss": BasicSequenceLoss,
    # "loss_params": {
    #     'offset_target_by_one': False,
    # },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "augmentation": data_aug,
        "dataset_files": dataset_files,
        "max_duration": 16.7,
        "shuffle": True,
        "usernn": True,
        "jointctcrnn": True,
        # "syn_ver": 3,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
        "usernn": True,
        "jointctcrnn": True,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "dataset_files": [
            "/data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
        "jointctcrnn": True,
    },
}
