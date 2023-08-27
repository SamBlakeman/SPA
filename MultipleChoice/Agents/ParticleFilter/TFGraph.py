import os
import numpy as np
import tensorflow as tf
import time
import tensorflow.contrib.slim as slim

class TFGraph(object):

    def __init__(self, directory, vgg16_npy_path, input_dim, rand_action_prob, num_actions):
        self.ti = 0
        self.directory = directory
        self.input_dim = input_dim
        self.rand_action_prob = rand_action_prob
        self.num_actions = num_actions

        self.data_dict = np.load(os.path.dirname(__file__) + vgg16_npy_path,
                                 encoding='latin1', allow_pickle=True).item()

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.X = tf.placeholder(tf.float32, shape=(None, self.input_dim[0], input_dim[1], input_dim[2]), name="X")
            self.y = tf.placeholder(tf.float32, shape=(None), name="y")
            self.attention_weights = tf.placeholder(tf.float32, shape=(None, 7, 7, 512), name="attention_weights")

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
                self.pool5_attention = slim.flatten(tf.multiply(self.pool5, self.attention_weights))

            ############################### Deep RL #################################

            with tf.variable_scope('value'):
                self.hidden1 = tf.stop_gradient(self.pool5_attention)
                self.hidden2 = tf.layers.dense(inputs=self.hidden1, units=512, activation=tf.nn.relu)
                self.hidden3 = tf.layers.dense(inputs=self.hidden2, units=256, activation=tf.nn.relu)
                self.value = slim.fully_connected(self.hidden3, 1, activation_fn=None)


            # Loss function
            with tf.name_scope("loss"):
                self.targets = tf.stop_gradient(self.y)
                self.loss = tf.reduce_mean(tf.square(self.targets - self.value), name='loss')

            # Minimizer
            self.learning_rate = 0.00025
            self.momentum = 0.95
            self.epsilon = 0.01

            with tf.name_scope("optimizer"):
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, momentum=self.momentum,
                                                           epsilon=self.epsilon)
                self.training_op = self.optimizer.minimize(self.loss)

            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

        self.data_dict = None

        return

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

    def GetFeatureMaps(self, input_images):

        return self.sess.run(self.pool5, feed_dict={self.X: input_images})

    def GetValues(self, input_images, attention_weights):

        values = self.sess.run(self.value, feed_dict={self.X: input_images,
                                                      self.attention_weights: attention_weights})

        return values

    def SelectAction(self, input_images, attention_weights):

        values = self.sess.run(self.value, feed_dict={self.X: input_images,
                                                      self.attention_weights: attention_weights})
        if(np.random.rand() > self.rand_action_prob):
            choice = np.argmax(values)
            bRand = False
        else:
            choice = np.random.randint(self.num_actions)
            bRand = True

        probs = 0

        return choice, probs, values, bRand

    def Train(self, prev_images, choice, reward, attention_weights):

        image = prev_images[choice, :, :, :]
        image = np.expand_dims(image, axis=0)

        attention = attention_weights[choice, :, :, :]
        attention = np.expand_dims(attention, axis=0)

        _, loss = self.sess.run([self.training_op, self.loss],
                      feed_dict={self.X: image, self.y: reward,
                                 self.attention_weights: attention})

        return loss

    def GetLosses(self, X, reward, attention_weights):

        rewards = np.ones(X.shape[0]) * reward

        loss = self.sess.run(self.loss,
                             feed_dict={self.X: X, self.y: rewards,
                                        self.attention_weights: attention_weights})

        return loss

    def SaveGraphAndVariables(self):
        save_path = self.saver.save(self.sess, self.directory)
        print('Model saved in ' + save_path)

        return

    def LoadGraphAndVariables(self):
        self.saver.restore(self.sess, self.directory)
        print('Model loaded from ' + self.directory)

        return


