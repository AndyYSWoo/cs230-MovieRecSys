import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from config import config

class DAEModel(object):
    def __init__(self, config):
        self.config = config

    def build(self):
        self.add_placeholders_op()
        self.build_autoencoder_op()
        self.add_loss_op()
        self.add_optimizer_op()
        # self.add_RMSE_op()
        self.add_r2_op()

    def initialize(self):
        self.sess = tf.Session()
        # TODO TensorBoard Monitoring or Logger
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_placeholders_op(self):
        self.input_placeholder = tf.placeholder(tf.float32, (None, self.N)) # Original ratings
        self.known_placeholder = tf.placeholder(tf.float32, (None, self.N)) # Known indices
        self.known_num_placeholder = tf.placeholder(tf.float32, ())         # Number of known ratings in batch
        self.mask_placeholder = tf.placeholder(tf.float32, (None, self.N))  # Masked indices
        self.meta_placeholder = tf.placeholder(tf.float32, (None, self.d)) if self.config.use_metadata else tf.placeholder(tf.float32) # ugly hack

    def build_autoencoder_op(self):
        scope = 'autoencoder'
        out = tf.multiply(self.input_placeholder, (1 - self.mask_placeholder))
        if self.config.use_metadata:
            out = tf.concat([out, self.meta_placeholder], axis=1)
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            for layer in range(config.n_layers):
                out = tf.contrib.layers.fully_connected(out, self.config.layer_size, activation_fn=tf.nn.tanh,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.lambd),
                                                        reuse=tf.AUTO_REUSE, scope=scope + '-l-'+str(layer))
        out = tf.contrib.layers.fully_connected(out, self.N, activation_fn=tf.nn.tanh,
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(self.config.lambd),
                                                        reuse=tf.AUTO_REUSE, scope=scope + '-l-out')
        self.prediction = out

    # problem on reduce mean
    def add_loss_op(self):
        self.loss = self.config.alpha * tf.reduce_sum(
            tf.square(tf.multiply(self.mask_placeholder,
                                  self.input_placeholder-self.prediction))) \
                    + self.config.beta * tf.reduce_sum(
            tf.square(tf.multiply(self.known_placeholder - self.mask_placeholder,
                                  self.input_placeholder-self.prediction)))
    # def add_RMSE_op(self):
    #     self.rmse =  tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.multiply(self.known_placeholder, self.input_placeholder),
    #                                                               tf.multiply(self.known_placeholder, self.prediction))))

    def add_r2_op(self):
        mean = tf.reduce_sum(tf.multiply(self.known_placeholder, self.input_placeholder)) / self.known_num_placeholder
        tot = tf.reduce_sum(tf.multiply(self.known_placeholder, tf.square(self.input_placeholder - mean)))
        res = tf.reduce_sum(tf.multiply(self.known_placeholder, tf.square(self.input_placeholder - self.prediction)))
        self.r2 = 1 - res / tot

    def add_optimizer_op(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr).minimize(self.loss)

    def train(self, R_train, S_train, K_train, M_train, R_dev, S_dev, K_dev, M_dev):
        data_size = R_train.shape[0]
        indices = np.arange(data_size)
        num_batches_per_epoch = 1 + data_size / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=num_batches_per_epoch * self.config.num_epochs)
        print 'Initial Dev R^2: {}'.format(self.evaluate_R2(R_dev, S_dev, K_dev, M_dev))
        for epoch in range(self.config.num_epochs):
            if self.config.shuffle:
                np.random.shuffle(indices)
            for minibatch_start in np.arange(0, data_size, self.config.batch_size):
                minibatch_indices = indices[minibatch_start:minibatch_start + self.config.batch_size]
                R_batch = R_train[minibatch_indices]
                S_batch = S_train[minibatch_indices] if self.config.use_metadata else None
                K_batch = K_train[minibatch_indices]
                num_known = np.count_nonzero(K_batch)
                M_batch = M_train[minibatch_indices]
                # training
                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.input_placeholder: R_batch,
                    self.meta_placeholder : S_batch,
                    self.known_placeholder: K_batch,
                    self.known_num_placeholder: num_known,
                    self.mask_placeholder : M_batch
                })
                prog.update(epoch * num_batches_per_epoch+1+minibatch_start/self.config.batch_size,
                            [('train loss', loss)])
                print ', Train R^2: {}'.format(self.evaluate_R2(R_train, S_train, K_train, M_train)),
            # eval on dev set
            print '\nepoch: {}, Dev R^2: {}'.format(epoch+1, self.evaluate_R2(R_dev, S_dev, K_dev, M_dev))

    # def evaluate_RMSE(self, R, S, K, M):
    #     rmse = self.sess.run([self.rmse], feed_dict={
    #         self.input_placeholder: R,
    #         self.meta_placeholder : S,
    #         self.known_placeholder: K,
    #         self.mask_placeholder : np.zeros(M.shape)  # no mask when calculating RMSE
    #     })
    #     return rmse

    def evaluate_R2(self, R, S, K, M):
        r2 = self.sess.run([self.r2], feed_dict={
            self.input_placeholder: R,
            self.meta_placeholder : S,
            self.known_placeholder: K,
            self.known_num_placeholder: np.count_nonzero(K),
            self.mask_placeholder : np.zeros(M.shape)  # no mask when calculating RMSE
        })
        return r2

    def load_data(self, rating_path):
        return pd.read_csv(rating_path, delimiter=' ', header=None).as_matrix()

    def load_metadata(self, metadata_path):
        meta_fp = open(metadata_path, 'rb')
        metadata_list = pickle.load(meta_fp)
        metadata = np.concatenate(metadata_list).reshape((len(metadata_list), metadata_list[0].shape[0]))
        return metadata

    def random_data(self):
        N = 6000 # 20133
        M = 4000 # 6672
        density = 0.13
        R = np.random.uniform(1, 5, (M, N)) * ((np.random.uniform(0, 1, (M,N)) < density).astype(int))
        return R

    def get_known_indices(self, R):
        return (R != 0).astype(int)

    def get_mask_indices(self, K):
        return (K * np.random.uniform(0, 1, K.shape) > (1-self.config.mask_ratio)).astype(int)

    # [1, 5] -> [-1, 1], -1.5 -> 0
    def normalize(self, R):
        R -= 3
        R /= 2.0
        R[R==-1.5] = 0

    def pca(self, R):
        u, s, v = np.linalg.svd(R.T)
        print 'svd done, u: {}, s:{}.'.format(u.shape, s.shape)
        tot = np.sum(s)
        cur = tot
        for k in range(len(s)-1, 1, -1):
            cur -= s[k]
            keep = cur / tot
            # print 'k={}, keep={}'.format(k, keep)
            if keep <= 0.95:
                print 'For k={}, more than {}% variance is kept.'.format(k+1, keep*100)
                break
        return k+1, u[:,:k+1]

    def run(self):
        R = self.load_data('../data/ratings-small')
        self.N = R.shape[1] # number of users
        self.M = R.shape[0] # number of movies

        K = self.get_known_indices(R)
        M = self.get_mask_indices(K)    # add noise mask
        self.normalize(R)

        if self.config.debug:
            print '# of Users: {}, # of Movies: {}'.format(self.N, self.M)
            k_c = np.count_nonzero(K)
            m_c = np.count_nonzero(M)
            print 'Density: {}, Known: {}, Mask: {}, Mask Ratio: {}'.format(float(k_c)/(self.N * self.M), k_c, m_c, float(m_c) / k_c)
        if self.config.use_pca:
            self.num_components, self.u_reduce = self.pca(R)
            print self.u_reduce.shape

        # split into train/dev/test sets
        train_size = int(self.M * 0.8)
        dev_size = int(self.M * 0.1)
        R_train, R_dev, R_test = np.split(R, [train_size, train_size + dev_size])
        K_train, K_dev, K_test = np.split(K, [train_size, train_size + dev_size])
        M_train, M_dev, M_test = np.split(M, [train_size, train_size + dev_size])
        if self.config.use_metadata:
            S = self.load_metadata('../data/overviewVectors')
            self.d = S.shape[1] # dimension of side infomation
            S_train, S_dev, S_test = np.split(S, [train_size, train_size + dev_size])
        else:
            S_train, S_dev, S_test = None, None, None
        self.M = R_train.shape[0]

        self.build()
        self.initialize()
        self.train(R_train, S_train, K_train, M_train, R_dev, S_dev, K_dev, M_dev)

        print 'R^2 on test set: {}'.format(self.evaluate_R2(R_test, S_test, K_test, M_test))


if __name__=='__main__':
    model = DAEModel(config)
    model.run()
