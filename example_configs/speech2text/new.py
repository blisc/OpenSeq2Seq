"convnet_layers": [
    {"repeat": 1, "kernel_size": [11], "stride": [2],
     "num_channels": 256, "dropout_keep_prob": 0.8, ...},
    
    {"repeat": 5, "kernel_size": [11],
     "num_channels": 256, "dropout_keep_prob": 0.8, ...},
    
    ...
    
    {"repeat": 5, "kernel_size": [25],
     "num_channels": 768, "dropout_keep_prob": 0.7, ...},
    
    {"repeat": 1, "kernel_size": [29], "dilation":[2],
     "num_channels": 896, "dropout_keep_prob": 0.6, ...},
    
    {"repeat": 1, "kernel_size": [1],
     "num_channels": 1024, "dropout_keep_prob": 0.6, ...}]}
