"convnet_layers": [
    {"type": "conv1d", "repeat": 1,
     "kernel_size": [11], "stride": [2],
     "num_channels": 256, "padding": "SAME",
     "dilation":[1], "dropout_keep_prob": 0.8},

    {"type": "conv1d", "repeat": 5,
     "kernel_size": [11], "stride": [1],
     "num_channels": 256, "padding": "SAME",
     "dilation":[1], "dropout_keep_prob": 0.8,
     "residual": True, "residual_dense": residual_dense},

    ...

    {"type": "conv1d", "repeat": 5,
     "kernel_size": [25], "stride": [1],
     "num_channels": 768, "padding": "SAME",
     "dilation":[1], "dropout_keep_prob": 0.7,
     "residual": True, "residual_dense": residual_dense},

    {"type": "conv1d", "repeat": 1,
     "kernel_size": [29], "stride": [1],
     "num_channels": 896, "padding": "SAME",
     "dilation":[2], "dropout_keep_prob": 0.6},

    {"type": "conv1d", "repeat": 1,
     "kernel_size": [1], "stride": [1],
     "num_channels": 1024, "padding": "SAME",
     "dilation":[1], "dropout_keep_prob": 0.6}],
