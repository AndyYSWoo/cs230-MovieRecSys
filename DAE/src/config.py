import tensorflow as tf

class config():
    tb_dir = '../tb/'
    debug = True

    use_pca = False

    # dataset
    mask_ratio = 0.25
    keep_variance = 0.99

    # metadata
    use_metadata = False

    # architecture hyperparameters
    n_layers    = 4                 # number of hidden layers
    layer_size  = 80               # number of nuerons per layer
    activation  = tf.nn.tanh        # hidden layers' activation

    # training hyperparameters
    num_epochs  = 500
    batch_size  = 2100
    lr          = 1e-3  # learning rate
    lambd       = 0   # l2-regularization
    alpha       = 1     # emphasis on denoising
    beta        = 0.5   # emphasis on recovering
    shuffle     = True  # shuffle training data every epoch
    dropout     = 0.8
    max_gradient_norm = 500000

    #config for score model
    lr_score = 1e-3
    batch_size_score = 210 
    n_layers_score = 4
    layer_size_score = 80
    vector_dimension = 50
    lambd_score = 0.1
    like_threshold = 0.5