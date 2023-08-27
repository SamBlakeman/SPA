import os
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.slim as slim
from scipy.signal import lfilter

class QGraph(object):
    def __init__(self, input_dim, num_actions, discount_factor, directory, vgg16_npy_path, attention_dim):

        self.ti = 0
        self.num_actions = num_actions
        self.directory = directory
        self.input_dim = input_dim
        self.beta_entropy = 0.01
        self.gamma = discount_factor
        self.attention_dim = attention_dim

        self.data_dict = np.load(os.path.dirname(__file__) + vgg16_npy_path,
                                 encoding='latin1', allow_pickle=True).item()
        print("npy file loaded")

        start_time = time.time()
        print("build model started")

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():

            self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim[0], input_dim[1], input_dim[2]), name="X")
            self.y = tf.placeholder(tf.float32, shape=(None), name="y")
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)

            self.actions_onehot = tf.one_hot(self.actions, self.num_actions, dtype=tf.float32)
            self.target_v = tf.placeholder(shape=[None], dtype=tf.float32)
            self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

            ############################### VGG16 #################################
            with tf.variable_scope('vgg16'):

                self.conv1_1 = self.conv_layer(self.X, "conv1_1")
                self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
                self.pool1 = self.max_pool(self.conv1_2, 'pool1')

                self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
                self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
                self.pool2 = self.max_pool(self.conv2_2, 'pool2')

                self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
                self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
                self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
                self.pool3 = self.max_pool(self.conv3_3, 'pool3')

                self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
                self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
                self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
                self.pool4 = self.max_pool(self.conv4_3, 'pool4')

                self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
                self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
                self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
                self.pool5 = self.max_pool(self.conv5_3, 'pool5')
                self.pool5_flat = slim.flatten(self.pool5)

            ############################### Self-Attention #################################

            with tf.variable_scope('self-attention'):
                self.queries = tf.expand_dims(tf.layers.dense(inputs=self.pool5_flat, units=self.attention_dim,
                                               activation=tf.nn.relu), axis=1)
                self.keys = tf.reshape(tf.layers.dense(inputs=self.pool5_flat, units=512 * self.attention_dim,
                                                       activation=tf.nn.relu), [-1, 512, self.attention_dim])
                # self.values = tf.reshape(tf.layers.dense(inputs=self.attention_input, units=512 * self.attention_dim,
                #                                          activation=tf.nn.relu), [512, self.attention_dim])

                self.attention_unscaled = tf.matmul(self.queries, self.keys, transpose_b=True)
                self.d_k = tf.cast(tf.shape(self.keys)[-1], dtype=tf.float32)
                self.attention_scaled = tf.divide(self.attention_unscaled, tf.sqrt(self.d_k))
                self.attention_weights = tf.tile(tf.expand_dims(tf.nn.softmax(
                    self.attention_scaled, dim=-1), axis=1), [1, 7, 7, 1])
                self.attention_output = tf.multiply(self.pool5, self.attention_weights)

            ############################### LSTM #################################

            with tf.variable_scope('ac'):

                self.hidden1 = slim.flatten(self.attention_output)
                self.hidden2 = tf.layers.dense(inputs=self.hidden1, units=1024, activation=tf.nn.relu)

                # Recurrent network for temporal dependencies
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)

                c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
                h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
                self.state_init = [c_init, h_init]

                self.c_in = tf.placeholder(tf.float32, [None, lstm_cell.state_size.c], name="c_in")
                self.h_in = tf.placeholder(tf.float32, [None, lstm_cell.state_size.h], name="h_in")

                rnn_in = tf.expand_dims(self.hidden2, [0])
                step_size = tf.shape(self.X)[:1]
                state_in = tf.contrib.rnn.LSTMStateTuple(self.c_in, self.h_in)

                lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                    lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                    time_major=False)

                lstm_c, lstm_h = lstm_state
                self.state_out = (lstm_c[:1, :], lstm_h[:1, :])
                rnn_out = tf.reshape(lstm_outputs, [-1, 256])

                # Output layers for policy and value estimations
                self.policy = slim.fully_connected(rnn_out, self.num_actions,
                                                   activation_fn=tf.nn.softmax)
                self.value = slim.fully_connected(rnn_out, 1,
                                                  activation_fn=None)

            # Loss functions
            self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, [1])
            self.value_loss = tf.reduce_mean(tf.square(self.target_v - tf.reshape(self.value, [-1])))
            self.entropy = -(-tf.reduce_sum(self.policy * tf.log(self.policy))) * self.beta_entropy
            self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs) * self.advantages)
            self.loss = self.value_loss + self.policy_loss + self.entropy

            ##########################################################################

            with tf.name_scope("optimizer"):
                self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
                local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ac')
                self.gradients = tf.gradients(self.loss, local_vars)
                self.var_norms = tf.global_norm(local_vars)
                grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
                self.apply_grads = self.optimizer.apply_gradients(zip(grads, local_vars))

            tf.add_to_collection('value', self.value)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        self.batch_rnn_state = self.state_init

        self.data_dict = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")

    def GetFeatureMaps(self, X):

        return self.sess.run(self.pool5, feed_dict={self.X: X})


    def SelectAction(self, X, rnn_state):

        a_dist, v, rnn_state = self.sess.run(
            [self.policy, self.value, self.state_out],
            feed_dict={self.X: X,
                       self.c_in: rnn_state[0],
                       self.h_in: rnn_state[1]})
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = int(np.argmax(a_dist == a))

        return a, v, rnn_state, a_dist

    def GetValues(self, X, rnn_state):

        v = self.sess.run(self.value,
                      feed_dict={self.X: X,
                                 self.c_in: rnn_state[0],
                                 self.h_in: rnn_state[1]})

        return v

    def GetTargets(self, rollout, bootstrap_value):
        rollout = np.array(rollout)
        rewards = rollout[:, 2]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]

        return discounted_rewards


    def Train(self, rollout, bootstrap_value):

        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]
        advantages = discounted_rewards - values

        feed_dict = {self.target_v: discounted_rewards,
                     self.X: np.vstack(observations),
                     self.actions: actions,
                     self.advantages: advantages,
                     self.c_in: self.batch_rnn_state[0],
                     self.h_in: self.batch_rnn_state[1]}

        _, self.batch_rnn_state, v_loss, p_loss, entropy = self.sess.run([self.apply_grads, self.state_out,
                                                 self.value_loss, self.policy_loss, self.entropy],
                                                feed_dict=feed_dict)


        return v_loss, p_loss, entropy

    def GetLosses(self, rollout, bootstrap_value):

        rollout = np.array(rollout)
        observations = rollout[:, 0]
        actions = rollout[:, 1]
        rewards = rollout[:, 2]
        next_observations = rollout[:, 3]
        values = rollout[:, 5]

        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = self.discount(self.rewards_plus)[:-1]
        advantages = discounted_rewards - values

        feed_dict = {self.target_v: discounted_rewards,
                     self.X: np.vstack(observations),
                     self.actions: actions,
                     self.advantages: advantages,
                     self.c_in: self.batch_rnn_state[0],
                     self.h_in: self.batch_rnn_state[1]}

        v_loss, p_loss = self.sess.run([self.value_loss, self.policy_loss], feed_dict=feed_dict)

        loss = np.abs(v_loss) #+ np.abs(p_loss)

        return loss

    def discount(self, x):
        return lfilter([1], [1, -self.gamma], x[::-1], axis=0)[::-1]


    def SaveGraphAndVariables(self):
        save_path = self.saver.save(self.sess, self.directory)
        print('Model saved in ' + save_path)

        return

    def LoadGraphAndVariables(self):
        self.saver.restore(self.sess, self.directory)
        print('Model loaded from ' + self.directory)

        return

    def Clear(self):
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(None, 1), name="X")
            self.init = tf.global_variables_initializer()
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)
        self.sess = tf.Session("")
        return
