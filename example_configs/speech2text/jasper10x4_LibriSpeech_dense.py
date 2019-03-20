# pylint: skip-file
import tensorflow as tf
from open_seq2seq.models import Speech2Text
from open_seq2seq.encoders.tdnn_dense_encoder import TDNNDenseEncoder
from open_seq2seq.decoders import FullyConnectedCTCDecoder
from open_seq2seq.data.speech2text.speech2text import Speech2TextDataLayer
from open_seq2seq.losses import CTCLoss
from open_seq2seq.optimizers.lr_policies import poly_decay

normalization = "batch_norm"
activation = tf.nn.relu
# gate_activation = None

mode = "dense" #"dense" or "denser"
repeat_1 = 4
repeat_2 = 4
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
    "num_epochs": 50,

    "num_gpus": 8,
    "batch_size_per_gpu": 32,
    "iter_size": 1,

    "save_summaries_steps": 100,
    "print_loss_steps": 10,
    "print_samples_steps": 2200,
    "eval_steps": 2200,
    "save_checkpoint_steps": 1100,
    "logdir": "nsr_log_folder",
    "num_checkpoints": 2,

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

    "encoder": TDNNDenseEncoder,
    "encoder_params": {
        "convnet_layers": [
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [11], "stride": [2],
                "num_channels": 128, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [11], "stride": [1],
                "num_channels": 200, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [11], "stride": [1],
                "num_channels": 200, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [13], "stride": [1],
                "num_channels": 280, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [13], "stride": [1],
                "num_channels": 280, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [17], "stride": [1],
                "num_channels": 344, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [17], "stride": [1],
                "num_channels": 344, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.8 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [21], "stride": [1],
                "num_channels": 424, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [21], "stride": [1],
                "num_channels": 424, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_1,
                "kernel_size": [25], "stride": [1],
                "num_channels": 480, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": repeat_2,
                "kernel_size": [25], "stride": [1],
                "num_channels": 480, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.7 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [29], "stride": [1],
                "num_channels": 560, "padding": "SAME",
                "dilation":[2], "dropout_keep_prob": 0.6 * dropout_factor,
            },
            {
                "type": "conv1d", "repeat": 1,
                "kernel_size": [1], "stride": [1],
                "num_channels": 640, "padding": "SAME",
                "dilation":[1], "dropout_keep_prob": 0.6 * dropout_factor,
            }
        ],

        "dropout_keep_prob": 0.7 * dropout_factor,

        "initializer": tf.contrib.layers.xavier_initializer,
        "initializer_params": {
            'uniform': False,
        },
        "normalization": normalization,
        "activation_fn": activation,
        "data_format": "channels_last",
        "use_conv_mask": True,
        "dense_mode": mode,
    },

    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {
        "initializer": tf.contrib.layers.xavier_initializer,
        "use_language_model": False,

        # params for decoding the sequence with language model
        "beam_width": 512,
        "alpha": 2.0,
        "beta": 1.5,

        "decoder_library_path": "ctc_decoder_with_lm/libctc_decoder_with_kenlm.so",
        "lm_path": "language_model/4-gram.binary",
        "trie_path": "language_model/trie.binary",
        "alphabet_config_path": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "infer_logits_to_pickle": False,
    },
    "loss": CTCLoss,
    "loss_params": {},

    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "num_audio_features": 64,
        "input_type": "logfbank",
        "vocab_file": "open_seq2seq/test_utils/toy_speech_data/vocab.txt",
        "norm_per_feature": True,
        "window_type": "hamming",
        "syn_enable": False,
        "syn_subdirs": ["1_50", "2_44", "3_47", "50", "46", "48"],
        "precompute_mel_basis": True,
        "sample_freq": 16000,
        "pad_to": 16
    },
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "augmentation": data_aug,
        "dataset_files": dataset_files,
        "max_duration": 16.7,
        "shuffle": True,
        "dither": 1e-5,
    },
}

eval_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
            "/data/librispeech/librivox-dev-clean.csv",
        ],
        "shuffle": False,
    },
}

infer_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "dataset_files": [
            "/data/librispeech/librivox-test-clean.csv",
        ],
        "shuffle": False,
    },
}