import tensorflow as tf

class config():
    debug = True

    # dataset
    mask_ratio = 0.25

    # metadata
    use_metadata = False

    # architecture hyperparameters
    n_layers    = 1                 # number of hidden layers
    layer_size  = 1500               # number of nuerons per layer
    activation  = tf.nn.relu        # hidden layers' activation

    # training hyperparameters
    num_epochs  = 100
    batch_size  = 500
    lr          = 1e-3
    lambd       = 0.0
    alpha       = 1
    beta        = 0.5
    shuffle     = True