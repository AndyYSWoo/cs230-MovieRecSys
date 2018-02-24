import tensorflow as tf

class config():
    # dataset TODO set this dynamically to model not in config
    N = 20000   # number of users
    M = 40000   # number of movies

    # metadata
    use_metadata = False

    # architecture hyperparameters
    n_layers = 1                # number of hidden layers
    layer_size = 750            # number of nuerons per layer
    activation = tf.nn.relu     # hidden layers' activation

    lr = 1e-3
    alpha = 1
    beta  = 0.5