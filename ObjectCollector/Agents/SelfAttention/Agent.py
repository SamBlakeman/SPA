import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from keras.applications.vgg16 import preprocess_input

from ObjectCollector.Agents.SelfAttention.QGraph import QGraph


class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        np.random.seed(env_params['random_seed'])
        self.directory = directory
        self.im_width = agent_params['im_width']
        self.im_height = agent_params['im_height']
        self.num_actions = env_params['num_actions']

        self.buffer_size = agent_params['buffer_size']  # buffer length for LSTM
        self.save_update = agent_params['save_update']
        self.frame_skip = agent_params['frame_skip']
        self.results = {'rewards': [], 'lengths': [], 'particle_states': [], 'attention': []}

        # Particle Filter Settings
        self.num_filters = 512

        self.save_counter = 0
        self.frame_counter = 0
        self.trial_reward = 0
        self.trial_length = 0
        self.plot_num = 0
        self.bStartBuffer = False

        self.episode_buffer = []
        self.sequence = []
        self.phi = np.array([])
        self.reward = 0
        self.action = 0

        self.q_graph = QGraph((self.im_height, self.im_width, 3), self.num_actions,
                              agent_params['discount_factor'], self.directory, '/../../../VGG16/vgg16.npy',
                              agent_params['self_attn_dim'])
        self.q_graph.SaveGraphAndVariables()

        self.prev_value = np.array([[0]])
        self.prev_action = 0
        self.prev_state = None
        self.prev_rnn_state = self.q_graph.state_init

        return

    def PreprocessSequence(self, pxarray):
        img = Image.fromarray(pxarray)
        img = img.resize([self.im_width, self.im_height])
        img_grey = img.convert('LA')
        img_grey = np.array(img_grey).astype(np.uint8)
        img = img_grey[:, :, 0]
        img = np.expand_dims(np.expand_dims(img, axis=-1), axis=0)
        img = np.tile(img, (1, 1, 1, 3))
        img = preprocess_input(img)
        return img

    def Update(self, reward, pxarray, bTrial_over):

        self.frame_counter += 1
        self.reward += reward

        if(self.frame_counter % self.frame_skip == 0 or bTrial_over):
            self.frame_counter = 0
            self.save_counter += 1

            self.state = self.PreprocessSequence(pxarray)
            self.RecordResults(bTrial_over, self.reward)

            # Q Graph Training
            if(bTrial_over):
                self.episode_buffer.append(
                    [self.prev_state, self.prev_action, self.reward, self.prev_state, bTrial_over, self.prev_value[0, 0]])

                self.v_loss, self.p_loss, self.entropy = self.q_graph.Train(self.episode_buffer, 0.0)
                self.episode_buffer = []
                self.prev_rnn_state = self.q_graph.state_init

                self.PrintDecisionValues()

            else:
                if(self.bStartBuffer):
                    self.episode_buffer.append(
                        [self.prev_state, self.prev_action, self.reward, self.state, bTrial_over, self.prev_value[0, 0]])

                    if(len(self.episode_buffer) == self.buffer_size):
                        bootstrap_value = self.q_graph.GetValues(self.state, self.prev_rnn_state)[0,0]

                        self.v_loss, self.p_loss, self.entropy = self.q_graph.Train(self.episode_buffer, bootstrap_value)

                        self.episode_buffer = []


            a, v, rnn_state, a_dist = self.q_graph.SelectAction(self.state, self.prev_rnn_state)

            self.action = a
            self.prev_value = v
            self.prev_action = np.copy(self.action)
            self.prev_state = np.copy(self.state)
            self.prev_rnn_state = rnn_state
            self.a_dist = a_dist

            self.bStartBuffer = True
            self.reward = 0

            if (self.save_counter >= self.save_update):
                self.save_counter = 0
                self.q_graph.SaveGraphAndVariables()
                self.PlotResults()

        return self.action

    def PrintEpisodeValues(self):

        print('     Episode Steps: ' + str(self.trial_length))
        print('     Episode Reward: ' + str(self.trial_reward))

        return

    def PrintDecisionValues(self):

        print('     v_loss: ' + str(self.v_loss))
        print('     p_loss: ' + str(self.p_loss))
        print('     entropy: ' + str(self.entropy))
        print('     v: ' + str(self.prev_value))
        print('     a: ' + str(self.action))
        print('     r: ' + str(self.reward))
        print('	    a_dist: ' + str(self.a_dist))

        return


    def RecordResults(self, bTrial_over, reward):

        self.trial_reward += reward
        self.trial_length += 1
        if (bTrial_over):
            self.PrintEpisodeValues()

            self.results['rewards'].append(self.trial_reward)
            self.trial_reward = 0

            self.results['lengths'].append(self.trial_length)
            self.trial_length = 0

        return


    def PlotResults(self):
        plt.switch_backend('agg')
        plt.figure()
        plt.plot(self.results['rewards'])
        plt.savefig(self.directory + 'AgentTrialRewards.pdf')
        plt.close()

        with open(self.directory + 'Results.pkl', 'wb') as handle:
            pickle.dump(self.results, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return

    def Clear(self):
        self.q_graph.Clear()
        return
