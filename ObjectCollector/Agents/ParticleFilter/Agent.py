import matplotlib.pyplot as plt
import numpy as np
import pickle
from PIL import Image
from keras.applications.vgg16 import preprocess_input

from ObjectCollector.Agents.ParticleFilter.QGraph import QGraph


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
        self.bAttention = agent_params['bAttention']
        self.bRand_attention = agent_params['bRand_attention']
        self.num_particles = agent_params['num_particles']
        self.tau = agent_params['tau']
        self.sigma = agent_params['sigma']

        if(not self.bAttention):
            self.attention = np.ones((1, 7, 7, self.num_filters))
        elif(self.bRand_attention):
            state = np.random.randint(0, 2, self.num_filters)
            state = np.expand_dims(np.expand_dims(np.expand_dims(state, axis=0), axis=0), axis=0)
            state = np.tile(state, (1, 7, 7, 1))
            self.attention = state / np.sum(state)
        else:
            self.attention = np.ones((1, 7, 7, self.num_filters)) / self.num_filters


        self.attention_update = agent_params['attention_update']
        self.ai = 0
        self.bottom_up_p = 0

        self.particle_states = []
        for p in range(self.num_particles):
            self.particle_states.append(np.ones((1, 7, 7, self.num_filters)) / self.num_filters)

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
                              agent_params['discount_factor'], self.directory, '/../../../VGG16/vgg16.npy')
        self.q_graph.SaveGraphAndVariables()

        self.prev_value = np.array([[0]])
        self.prev_action = 0
        self.prev_state = None
        self.prev_rnn_state = self.q_graph.state_init

        self.attention_history = []

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

                ######################################################################
                if (self.bAttention and not self.bRand_attention):
                    self.ai += 1
                    if (self.ai >= self.attention_update):
                        self.ai = 0
                        self.Movement()
                        self.Observe(0.0)
                        self.SetAttention()
                ######################################################################

                self.v_loss, self.p_loss, self.entropy = self.q_graph.Train(self.episode_buffer, 0.0,
                                                                            np.tile(self.attention, (len(self.episode_buffer), 1, 1, 1)))
                self.episode_buffer = []
                self.prev_rnn_state = self.q_graph.state_init

                self.PlotAttention()
                self.PrintDecisionValues()

            else:
                if(self.bStartBuffer):
                    self.episode_buffer.append(
                        [self.prev_state, self.prev_action, self.reward, self.state, bTrial_over, self.prev_value[0, 0]])

                    if(len(self.episode_buffer) == self.buffer_size):
                        bootstrap_value = self.q_graph.GetValues(self.state, self.prev_rnn_state, self.attention)[0,0]

                        ######################################################################
                        if (self.bAttention and not self.bRand_attention):
                            self.ai += 1
                            if (self.ai >= self.attention_update):
                                self.ai = 0
                                self.Movement()
                                self.Observe(bootstrap_value)
                                self.SetAttention()

                        ######################################################################

                        self.v_loss, self.p_loss, self.entropy = self.q_graph.Train(self.episode_buffer, bootstrap_value,
                                                                                    np.tile(self.attention, (len(self.episode_buffer), 1, 1, 1)))

                        self.episode_buffer = []


            a, v, rnn_state, a_dist = self.q_graph.SelectAction(self.state, self.prev_rnn_state, self.attention)

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


    def Observe(self, bootstrap_value):

        # Only do processing on unique states
        particle_states, counts = np.unique(np.array(self.particle_states), return_counts=True, axis=0)

        particle_states = particle_states / np.expand_dims(np.sum(particle_states, axis=1), axis=1)
        particle_states = np.expand_dims(np.expand_dims(np.array(particle_states), axis=1), axis=1)
        particle_states = np.tile(particle_states, (1, 7, 7, 1))

        # Weight particles based on error value
        losses = []
        for particle_state in [particle_states[i, :, :, :] for i in range(particle_states.shape[0])]:
            losses.append(self.q_graph.GetLosses(self.episode_buffer, bootstrap_value,
                                                 np.tile(np.expand_dims(particle_state, axis=0),
                                                         (len(self.episode_buffer), 1, 1, 1))))

        losses = np.array(losses)
        losses = losses - np.amin(losses)
        weights = np.exp(-losses * self.sigma) * counts
        weights /= np.sum(weights)

        # RE-SAMPLE
        particle_states, counts = np.unique(np.array(self.particle_states), return_counts=True, axis=0)
        inds = np.random.choice(a=np.arange(particle_states.shape[0]), size=self.num_particles,
                                p=weights, replace=True)
        self.particle_states = particle_states[inds, :]

        print('Performed Observe Step')
        print('Max Weight: ' + str(np.amax(weights)))
        print('Num Unique Particles: ' + str(counts.shape[0]))

        return

    def Movement(self):

        # Calculate probabilities based on bottom-up attention
        feature_maps = self.q_graph.GetFeatureMaps(np.vstack(np.array(self.episode_buffer)[:, 0]))
        feature_maps = np.reshape(feature_maps, (feature_maps.shape[0],
                                                 -1, feature_maps.shape[-1]))
        feature_maps = np.mean(feature_maps, axis=1)
        feature_maps /= np.expand_dims(np.sum(feature_maps, axis=1), axis=1)

        probs = np.mean(np.exp(feature_maps * self.tau), axis=0)
        probs /= np.amax(probs)

        for i in range(len(self.particle_states)):
            if (np.random.rand() > self.bottom_up_p):
                self.particle_states[i] = (np.random.rand(self.num_filters) < probs).astype('int')

        self.bottom_up_p = .99

        return

    def SetAttention(self):

        attention = np.mean(self.particle_states, axis=0)
        attention = attention / np.sum(attention)

        attention_expanded = np.expand_dims(np.expand_dims(np.expand_dims(attention, axis=0), axis=0), axis=0)
        self.attention = np.tile(attention_expanded, (1, 7, 7, 1))

        return

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

    def PlotAttention(self):

        self.attention_history.append(self.attention[0, 0, 0, :])
        _, counts = np.unique(np.array(self.particle_states), return_counts=True, axis=0)

        plt.switch_backend('agg')
        plt.figure()
        plt.imshow(np.array(self.attention_history), cmap='hot')
        plt.suptitle('Number of Unique States: ' + str(counts.shape[0]))
        plt.savefig(self.directory + 'AttentionHistory.pdf')
        plt.close()

        return

    def Clear(self):
        return
