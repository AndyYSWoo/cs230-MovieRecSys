import tensorflow as tf

class config():
    debug = True

    use_pca = True

    # dataset
    mask_ratio = 0.25
    keep_variance = 0.99

    # metadata
    use_metadata = False

    # architecture hyperparameters
    n_layers    = 1                 # number of hidden layers
    layer_size  = 500               # number of nuerons per layer
    activation  = tf.nn.tanh        # hidden layers' activation

    # training hyperparameters
    num_epochs  = 50
    batch_size  = 200
    lr          = 1e-3  # learning rate
    lambd       = 0.1   # l2-regularization
    alpha       = 1     # emphasis on denoising
    beta        = 0.5   # emphasis on recovering
    shuffle     = True  # shuffle training data every epoch
    dropout     = 1

    #config for score model
    n_layers_score = 4
    layer_size_score = 80
    vector_dimension = 50
    like_threshold = 0.5