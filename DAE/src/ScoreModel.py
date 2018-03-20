import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from config import config

class ScoreModel(object):
    def __init__(self, config):
        self.config = config

    def build(self):
        self.add_placeholders_op()
        self.build_prediction_op()
        self.add_loss_op()
        self.add_optimizer_op()
        self.add_Accurancy_op()

    def initialize(self):
        self.sess = tf.Session()
        # TODO TensorBoard Monitoring
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_placeholders_op(self):
        self.movie_placeholder = tf.placeholder(tf.int32, (None, 1)) # Original ratings
        self.user_placeholder = tf.placeholder(tf.int32, (None, 1)) # Known indices
        self.labels_placeholder = tf.placeholder(tf.int32, (None, 1))
        #self.meta_placeholder = tf.placeholder(tf.float32, (None, self.d)) if self.config.use_metadata else tf.placeholder(tf.float32) 
    
    def build_prediction_op(self):
        with tf.variable_scope("ScoreModel", reuse=tf.AUTO_REUSE):
            movieEmbeddings = tf.get_variable(name = "movieEmbeddings", shape = [self.M, self.config.vector_dimension], initializer=tf.random_normal_initializer(mean=0, stddev=1))  
            userEmbeddings = tf.get_variable(name = "userEmbeddings", shape = [self.N, self.config.vector_dimension], initializer=tf.random_normal_initializer(mean=0, stddev=1))  

        movieEmbed = tf.nn.embedding_lookup(movieEmbeddings, self.movie_placeholder)
        userEmbed = tf.nn.embedding_lookup(userEmbeddings, self.user_placeholder)

        userEmbedT = tf.transpose(userEmbed, perm=[0,2,1])
        product = tf.matmul(movieEmbed, userEmbedT)
        out = tf.nn.sigmoid(product)
        out = tf.squeeze(out, [1])
        self.prediction = out

    def add_loss_op(self):
        transformedLabels = tf.to_double(self.labels_placeholder) * 0.5 + 0.5
        transformedLabels = tf.cast(transformedLabels, tf.float32)
        self.loss = tf.losses.log_loss(labels = transformedLabels, predictions = self.prediction)

    def add_Accurancy_op(self):
        predictions = tf.cast((self.prediction >= 0.8), dtype = tf.int32)
        predictions = predictions * 2 - 1
        product = tf.multiply(predictions, self.labels_placeholder)
        product = product - 1
        nonzeros = tf.cast(tf.count_nonzero(product), tf.float32)
        self.accurancy = 1.0 - (nonzeros / tf.cast(tf.size(predictions), tf.float32))

    def add_optimizer_op(self):
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.config.lr_score).minimize(self.loss)

    def train(self, m_train, u_train, l_train, m_dev, u_dev, l_dev):
        
        #data_size = tf.cast(data_size, tf.float32)
        indices = np.arange(self.train_size)
        num_batches_per_epoch = 1 + self.train_size / self.config.batch_size
        prog = tf.keras.utils.Progbar(target=num_batches_per_epoch * self.config.num_epochs)
        print 'Initial Dev Accurance: {}'.format(self.evaluate_Accurancy(m_dev, u_dev, l_dev))
        start = 0
        for epoch in range(self.config.num_epochs):
            if self.config.shuffle:
                np.random.shuffle(indices)

            for minibatch_start in np.arange(0, self.train_size, self.config.batch_size):
                minibatch_indices = indices[minibatch_start:minibatch_start + self.config.batch_size]
                m_batch = m_train[minibatch_indices]
                u_batch = u_train[minibatch_indices] 
                l_batch = l_train[minibatch_indices]

                _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                    self.movie_placeholder: m_batch,
                    self.user_placeholder : u_batch,
                    self.labels_placeholder: l_batch,
                })
                #print('\nloss: {}'.format(loss))
                prog.update(epoch * num_batches_per_epoch+1+minibatch_start/self.config.batch_size,
                            [('train loss', loss)])

            start = 1

            # eval on dev set
            train_accurancy = self.evaluate_Accurancy(m_train, u_train, l_train)
            print '\nAccurancy on train set: {}'.format(train_accurancy)
            dev_accurancy = self.evaluate_Accurancy(m_dev, u_dev, l_dev)
            print '\nepoch: {}, Dev Accurancy: {}'.format(epoch+1, dev_accurancy)

    def evaluate_Accurancy(self, m, u, l):
        accurancy = self.sess.run([self.accurancy], feed_dict={
            self.movie_placeholder: m,
            self.user_placeholder : u,
            self.labels_placeholder: l
        })
        return accurancy

    def load_data(self, rating_path):
        return pd.read_csv(rating_path, delimiter=' ', header=None).as_matrix()

    def load_movieArray(self, movie_path):
        return np.loadtxt(movie_path)

    def load_userArray(self, user_path):
        return np.loadtxt(user_path)

    def load_labelArray(self, label_path):
        return np.loadtxt(label_path)

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

    def run(self):
        R = self.load_data('../data/ratings-binary.gz')
        self.N = R.shape[1] # number of users
        self.M = R.shape[0] # number of movies
        movieArray = self.load_movieArray('../data/movieArray-binary.gz')
        self.data_size = len(movieArray)
        userArray = self.load_movieArray('../data/userArray-binary.gz')
        labelArray = self.load_movieArray('../data/labelArray-binary.gz')

        if self.config.debug:
            print '# of Users: {}, # of Movies: {}'.format(self.N, self.M)
            k_c = self.data_size
            print 'Density: {}, Known: {}'.format(float(k_c)/(self.N * self.M), k_c)

        # split into train/dev/test sets
        # split into train/dev/test sets
        self.train_size = int(self.data_size * 0.8)
        self.dev_size = int(self.data_size * 0.1)

        indices = np.arange(self.data_size)
        if self.config.shuffle:
            np.random.shuffle(indices)

        trainIndex = indices[0 : self.train_size]
        devIndex = indices[self.train_size : self.train_size + self.dev_size]
        testIndex = indices[self.train_size + self.dev_size:]
        
        m_train = movieArray[trainIndex]
        m_dev = movieArray[devIndex]
        m_test = movieArray[testIndex]

        u_train = userArray[trainIndex]
        u_dev = userArray[devIndex]
        u_test = userArray[testIndex]

        l_train = labelArray[trainIndex]
        l_dev = labelArray[devIndex]
        l_test = labelArray[testIndex]

        m_train = np.reshape(m_train, (m_train.shape[0], 1))
        m_dev = np.reshape(m_dev, (m_dev.shape[0], 1))
        m_test = np.reshape(m_test, (m_test.shape[0], 1))

        u_train = np.reshape(u_train, (u_train.shape[0], 1))
        u_dev = np.reshape(u_dev, (u_dev.shape[0], 1))
        u_test = np.reshape(u_test, (u_test.shape[0], 1))

        l_train = np.reshape(l_train, (l_train.shape[0], 1))
        l_dev = np.reshape(l_dev, (l_dev.shape[0], 1))
        l_test = np.reshape(l_test, (l_test.shape[0], 1))

        #if self.config.use_metadata:
            #S = self.load_metadata('../data/overviewVectors')
            #self.d = S.shape[1] # dimension of side infomation
            #S_train, S_dev, S_test = np.split(S, [train_size, train_size + dev_size])
        #else:
            #S_train, S_dev, S_test = None, None, None
        #self.M = R_train.shape[0]

        self.build()
        self.initialize()
        self.train(m_train, u_train, l_train, m_dev, u_dev, l_dev)

        test_accurancy = self.evaluate_Accurancy(m_test, u_test, l_test)
        print 'Accurancy on test set: {}'.format(test_accurancy)


if __name__=='__main__':
    model = ScoreModel(config)
    model.run()
