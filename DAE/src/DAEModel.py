import numpy as np
import tensorflow as tf
from config import config

class DAEModel(object):
    def __init__(self, config):
        self.config = config

    def build(self):
        self.add_placeholders_op()
        self.build_autoencoder_op()
        self.add_loss_op()
        self.add_optimizer_op()

    def initialize(self):
        self.sess = tf.Session()
        # TODO TensorBoard Monitoring
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_placeholders_op(self):
        self.input_placeholder = tf.placeholder(tf.float32)
        # TODO do this in TF or NP?
        self.known_placeholder = tf.placeholder(tf.float32) # K
        self.mask_placeholder = tf.placeholder(tf.float32)  # M

    def build_autoencoder_op(self):
        scope = 'autoencoder'
        out = self.input_placeholder * (1 - self.mask_placeholder)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for layer in range(config.n_layers):
                out = tf.contrib.layers.fully_connected(out, self.config.layer_size, activation_fn=self.config.activation,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer,
                                                        reuse=tf.AUTO_REUSE, scope=scope + '-l['+str(layer) +']')
        out = tf.contrib.layers.fully_connected(out, self.N, activation_fn=tf.nn.sigmoid,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer,
                                                        reuse=tf.AUTO_REUSE, scope=scope + '-l[out]')
        self.prediction = out

    def add_loss_op(self):
        self.loss = self.config.alpha * tf.losses.mean_squared_error(self.mask_placeholder*self.input_placeholder
                                                 , self.mask_placeholder*self.prediction) \
                    + self.config.beta * tf.losses.mean_squared_error((self.known_placeholder - self.mask_placeholder) * self.input_placeholder,
                                                                      (self.known_placeholder - self.mask_placeholder) * self.prediction)

    def add_optimizer_op(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)

    def train(self):
        pass

    def evaluate(self):
        pass

    def run(self):
        # TODO load data & set N, M accordingly

        self.N = self.config.N
        self.M = self.config.M
        self.build()
        self.initialize()
        self.train()


if __name__=='__main__':
    model = DAEModel(config)
    model.run()
