import tensorflow as tf


class SE_GCN(object):
    def __init__(self, args, W2V, sent_max_len, vector_size, nfeat_v, nfeat_g,
                 nhid_vfeat, nhid_siamese, dropout_vfeat, dropout_siamese, nhid_final,
                 v_feature_shape, g_feature_shape):

        self.args = args
        self.embeddings = W2V

        self.step_size = sent_max_len
        self.vector_size = vector_size
        self.nfeat_v = nfeat_v
        self.nfeat_g = nfeat_g
        self.nhid_vfeat = nhid_vfeat
        self.nhid_siamese = nhid_siamese
        self.dropout_vfeat = dropout_vfeat
        self.dropout_siamese = dropout_siamese
        self.nhid_final = nhid_final

        self.window_size = 1
        self.num_filter = 32
        # elf.step_size = sent_max_len
        self.transformed_encoding_size = 2 * self.num_filter

        # batch * max_len, cig number of nodes, left right sentences idxs
        self.input_l = tf.placeholder(
            tf.int32, [None, self.step_size], name="input_xl")
        self.input_r = tf.placeholder(
            tf.int32, [None, self.step_size], name="input_xr")
        # vertices matrix (numofnodes, v_features)
        self.v_features = tf.placeholder(tf.float32, [None, v_feature_shape])

        # global graph features (1, g_features)
        self.g_feature = tf.placeholder(tf.float32, [g_feature_shape])
        # vertices (1, betweenness + pagerank + katz)
        print(self.g_feature.shape)
        # self.g_vertice = tf.placeholder(tf.float32, [g_vertices_shape])
        # print(self.g_vertice.shape)
        # label
        self.input_y = tf.placeholder(
            tf.float32, [1], name="input_y")
        # adj matrix  ( num of nodes, num of nodes)
        self.adj = tf.placeholder(tf.float32, [None, None])

        self.use_cig = (self.args.use_vfeatures or self.args.use_siamese)


        self.W = tf.get_variable(name='W', shape=self.embeddings.shape,
                                     initializer=tf.constant_initializer(self.embeddings), trainable=False)
        self.main_network()

        # embedding encoding

    def main_network(self):
        if self.use_cig:
            if self.args.use_siamese:
                x_l = self.get_embedding(self.input_l)
                x_r = self.get_embedding(self.input_r)

                batch, seq, embed = x_l.shape
                x_l = tf.expand_dims(x_l, -1)
                x_r = tf.expand_dims(x_r, -1)


                # x_l = tf.stop_gradient(x_l)
                # x_r = tf.stop_gradient(x_r)
                print(x_l.shape)
                print(x_r.shape)
                x_l = self.encoder_cnn(x_l)
                x_r = self.encoder_cnn(x_r)

                print(x_l.shape)
                print(x_r.shape)
                x_l = tf.reshape(x_l, [-1, self.num_filter])
                x_r = tf.reshape(x_r, [-1, self.num_filter])
                x_mul = tf.multiply(x_l, x_r)
                x_diff = tf.abs(tf.add(x_l, -x_r))
                self.x_siamese = tf.concat([x_mul, x_diff], 1)
                print("x_siame: ", self.x_siamese.shape)
            if self.args.use_gcn and self.args.use_siamese:
                for n_l in range(self.args.num_gcn_layers):
                    self.x_siamese = self.graph_conv(self.x_siamese, self.adj)
                    if n_l < self.args.num_gcn_layers - 1:
                        self.x_siamese = tf.nn.relu(self.x_siamese)
                        self.x_siamese = tf.nn.dropout(self.x_siamese, keep_prob=1.0-self.dropout_siamese)
            self.x_vfeat = self.v_features
            print("x_feat:", self.x_vfeat)
            if self.args.use_gcn and self.args.use_vfeatures:
                for n_l in range(self.args.num_gcn_layers):
                    self.x_vfeat = self.graph_conv(self.x_vfeat, self.adj)
                    if n_l < self.args.num_gcn_layers - 1:
                        self.x_vfeat = tf.nn.relu(self.x_vfeat)
                        self.x_vfeat = tf.nn.dropout(self.x_vfeat, keep_prob=1.0-self.dropout_vfeat)

            if self.args.use_vfeatures and not self.args.use_siamese:
                self.x = self.x_vfeat
            if not self.args.use_vfeatures and self.args.use_siamese:
                self.x = self.x_siamese
            if self.args.use_vfeatures and self.args.use_siamese:
                self.x = tf.concat([self.x_siamese, self.x_vfeat], 1)

            self.x = tf.reduce_mean(self.x, axis=0)
            print(self.x.shape)

            if self.args.use_gfeatures:
                self.x = tf.concat([self.x, self.g_feature],axis=0)

        else:
            self.x = self.g_feature

        print(self.x.shape)
        print(self.regressor())
        # regress_input_dim = self.regressor()
        W1 = tf.Variable(tf.random_normal([self.regressor(), self.nhid_final], stddev=0.1))
        b1 = tf.Variable(tf.random_normal([self.nhid_final], stddev=0.1))

        W2 = tf.Variable(tf.random_normal([self.nhid_final, 1], stddev=0.1))
        b2 = tf.Variable(tf.random_normal([1], stddev=0.1))

        self.out = tf.nn.relu(tf.tensordot(self.x, W1, axes=1)+b1)
        self.score = tf.nn.sigmoid(tf.tensordot(self.out, W2, axes=1)+b2)
        #self.score = tf.tensordot(self.out, W2, axes=1)+b2
        # self.out = tf.layers.Dense(self.x, units=self.nhid_final, activation=tf.nn.relu)
        # self.score = tf.layers.Dense(self.out, units=1, activation=tf.nn.sigmoid)


        #self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.score)
        self.loss = -tf.reduce_mean(tf.reduce_sum(self.input_y * tf.log(self.score+1e-8)+(1-self.input_y)*tf.log(1-self.score+1e-8)))

        return self.loss

    def get_embedding(self, input_x):
        # embedding
        embedded_words = tf.nn.embedding_lookup(self.W, input_x)
        return embedded_words

    def encoder_cnn(self, inputs):
        #with tf.name_scope("cnn_encoder"), tf.variable_scope('cnn_encoder'):
            # self.encoder = tf.nn.conv2d(input=inputs,
            #                             filter=[self.window_size, self.vector_size, 1, self.num_filter],
            #                             strides=[1, 1, 1, 1],
            #                             padding='VALID',
            #                             data_format='NCHW')
        self.encoder = tf.layers.conv2d(inputs=inputs,
                                        filters=self.num_filter,
                                        kernel_size=[self.window_size, self.vector_size],)
                                        #data_format='channels_first')

        self.encoder = tf.nn.relu(self.encoder)
        #print(self.encoder)
        self.encoder = tf.layers.max_pooling2d(self.encoder,
                                      pool_size=1,
                                      strides=self.step_size - self.window_size + 1,)
                                      #data_format='channels_first')

        return self.encoder


                # self.gcn_input_dim = self.transformed_encoding_size

    def graph_conv(self, inputs, adj):

        gcn_input_dim = inputs.shape[1].value

        W1 = tf.Variable(tf.random_normal([gcn_input_dim, self.nhid_vfeat], stddev=0.1))
        b1 = tf.Variable(tf.random_normal([self.nhid_vfeat], stddev=0.1))

        self.support1 = tf.tensordot(inputs, W1, 1)
        h1 = tf.tensordot(adj, self.support1, 1) + b1

        return h1

    def regressor(self):
        # gcn_input_dim = self.transformed_encoding_size
        regressor_input_dim = 0
        if self.args.use_gcn:
            if self.args.use_siamese and not self.args.use_vfeatures:
                regressor_input_dim = self.nhid_final
            if not self.args.use_siamese and self.args.use_vfeatures:
                regressor_input_dim = self.nhid_final
            if self.args.use_siamese and self.args.use_vfeatures:
                regressor_input_dim = 2 * self.nhid_final
            if not self.args.use_siamese and not self.args.use_vfeatures:
                regressor_input_dim = 0
            if self.args.use_gfeatures:
                regressor_input_dim += self.nfeat_g
        else:
            if self.args.use_siamese and not self.args.use_vfeatures:
                regressor_input_dim = self.transformed_encoding_size
            if not self.args.use_siamese and self.args.use_vfeatures:
                regressor_input_dim = self.nfeat_v
            if self.args.use_siamese and self.args.use_vfeatures:
                regressor_input_dim = self.transformed_encoding_size + self.nfeat_v
            if not self.args.use_siamese and not self.args.use_vfeatures:
                regressor_input_dim = 0
            if self.args.use_gfeatures:
                regressor_input_dim += self.nfeat_g
        return regressor_input_dim
