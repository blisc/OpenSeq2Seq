base_model = Speech2Text
base_params = {
    "num_epochs": 400,
    "num_gpus": 8,
    "batch_size_per_gpu": 32,
    
    "optimizer": NovoGrad,
    "optimizer_params": {...},
    "lr_policy": poly_decay,
    "lr_policy_params": {...},
    "loss": CTCLoss,
    "loss_params": {...},
    "decoder": FullyConnectedCTCDecoder,
    "decoder_params": {...},
    "encoder": TDNNEncoder,
    "encoder_params": {...},
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {...},
}

train_params = {
    "data_layer": Speech2TextDataLayer,
    "data_layer_params": {
        "augmentation": {
            'speed_perturbation_ratio': [-1.10, 1., 1.10],
        },
        "dataset_files": [
            "/data/librispeech/librivox-train-clean-100.csv",
            "/data/librispeech/librivox-train-clean-360.csv",
            "/data/librispeech/librivox-train-other-500.csv"
        ],
        "max_duration": 16.7,
        "shuffle": True,
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
            "/mnt/hdd/data/Librispeech/librispeech/librivox-dev-clean-2.csv",
        ],
        "shuffle": False,
    },
}
