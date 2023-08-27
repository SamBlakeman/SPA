import numpy as np
import matplotlib.pyplot as plt

from MultipleChoice.Agents.ParticleFilter.TFGraph import TFGraph
from keras.applications.vgg16 import preprocess_input

class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.num_filters = 512
        self.bAttention = agent_params['bAttention']
        self.num_particles = agent_params['num_particles']
        self.tau = agent_params['tau']
        self.sigma = agent_params['sigma']
        self.bBottom_up = agent_params['bBottom_up']
        self.results = {'particle_states': [], 'attention': []}
        self.bottom_up_p = 0

        self.attention = np.ones((3, 7, 7, self.num_filters)) / self.num_filters

        self.particle_states = []
        for p in range(self.num_particles):
            self.particle_states.append(np.ones((1, 7, 7, self.num_filters)) / self.num_filters)

        self.directory = directory
        self.bVerbose = agent_params['bVerbose']
        self.rand_action_prob = agent_params['rand_action_prob']
        np.random.seed(env_params['random_seed'])

        self.num_actions = len(env_params['object101_categories'])
        self.choice = 0
        self.prev_value = 0
        self.prev_feature_maps = False
        self.bStart_learning = False
        self.bRand = False
        self.trial_num = 0

        self.TFGraph = TFGraph(directory=self.directory, vgg16_npy_path='/../../../VGG16/vgg16.npy',
                               input_dim=(env_params['image_height'], env_params['image_width'], 3),
                               rand_action_prob=self.rand_action_prob, num_actions=self.num_actions)

        self.attention_history = []
        self.average_reward = 0
        self.n = 0

        self.block_size = env_params['num_training_trials']

        return

    def Update(self, trial_images, reward, regime):

        loss = 0
        self.n += 1
        self.average_reward = (self.average_reward * (self.n-1) + reward) / self.n

        self.trial_num += 1
        trial_images = preprocess_input(np.array(trial_images))

        if(self.bStart_learning):

            if (self.bAttention):
                feature_maps = self.Movement()
                counts, weights = self.Observe(reward)
                self.SetAttention(regime=regime)

                self.prev_feature_maps = feature_maps

            if(regime == 'training'):
                loss = self.TFGraph.Train(prev_images=self.prev_images, choice=self.choice,
                                   reward=reward, attention_weights=self.attention)
                self.bottom_up_p = .99


        self.choice, probs, values, self.bRand = self.TFGraph.SelectAction(input_images=trial_images,
                                                attention_weights=self.attention)
        self.prev_images = trial_images
        self.prev_value = values[self.choice]

        if(self.bVerbose and self.trial_num % self.block_size == 0):
            print('Trial: ' + str(self.trial_num))
            print('    Average reward: ' + str(self.average_reward))
            print('    Values: ' + str(values))
            print('    Loss: ' + str(loss))
            if(self.bAttention):
                print('    Number of unique states: ' + str(counts.shape[0]))
                print('    Max weight: ' + str(np.amax(weights)))
                print('    Bottom up p: ' + str(self.bottom_up_p))
            self.n = 0
            self.average_reward = 0

        if(not self.bStart_learning):
            self.bStart_learning = True

        self.PlotAttention()

        return self.choice

    def Movement(self):

        if(self.bBottom_up):
            feature_maps = self.TFGraph.GetFeatureMaps(self.prev_images)
            feature_maps = np.reshape(feature_maps, (feature_maps.shape[0],
                                                     -1, feature_maps.shape[-1]))
            feature_maps = np.mean(feature_maps, axis=1)
            feature_maps /= np.expand_dims(np.sum(feature_maps, axis=1), axis=1)

            probs = np.exp(feature_maps * self.tau)
        else:
            feature_maps = None

        for i in range(len(self.particle_states)):
            if (np.random.rand() > self.bottom_up_p):
                if(self.bBottom_up):
                    p = np.random.randint(probs.shape[0])
                    image_probs = probs[p, :]
                    image_probs /= np.amax(image_probs)
                else:
                    image_probs = np.random.randint(0, 2, self.num_filters)

                self.particle_states[i] = (np.random.rand(self.num_filters) < image_probs).astype('int')

        return feature_maps


    def Observe(self, reward):

        # Only do processing on unique states
        particle_states, counts = np.unique(np.array(self.particle_states), return_counts=True, axis=0)

        particle_states = particle_states / np.expand_dims(np.sum(particle_states, axis=1), axis=1)
        particle_states = np.expand_dims(np.expand_dims(np.array(particle_states), axis=1), axis=1)
        particle_states = np.tile(particle_states, (1, 7, 7, 1))

        image = np.expand_dims(self.prev_images[self.choice, :, :, :], axis=0)

        losses = []
        for particle_state in [particle_states[i, :, :, :] for i in range(particle_states.shape[0])]:
            losses.append(self.TFGraph.GetLosses(image, reward, np.expand_dims(particle_state, axis=0)))

        losses = np.array(losses)
        losses = losses - np.amin(losses)
        weights = np.exp(-losses * self.sigma) * counts
        weights /= np.sum(weights)

        # RE-SAMPLE
        particle_states, counts = np.unique(np.array(self.particle_states), return_counts=True, axis=0)
        inds = np.random.choice(a=np.arange(particle_states.shape[0]), size=self.num_particles,
                                p=weights, replace=True)
        self.particle_states = particle_states[inds, :]

        return counts, weights

    def SetAttention(self, regime):

        attention = np.mean(self.particle_states, axis=0)
        attention = attention / np.sum(attention)

        attention_expanded = np.expand_dims(np.expand_dims(np.expand_dims(attention, axis=0), axis=0), axis=0)
        self.attention = np.tile(attention_expanded, (3, 7, 7, 1))

        if (regime == 'test'):
            self.results['particle_states'].append(np.array(self.particle_states))
            self.results['attention'].append(attention)

        return


    def PlotAttention(self):

        # self.attention_history.append(self.attention[0, 0, 0, :])
        # _, counts = np.unique(np.array(self.particle_states), return_counts=True, axis=0)
        #
        # plt.switch_backend('agg')
        # plt.figure()
        # plt.imshow(np.array(self.attention_history), cmap='hot')
        # plt.suptitle('Number of Unique States: ' + str(counts.shape[0]))
        # plt.savefig(self.directory + 'AttentionHistory.pdf')
        # plt.close()

        return


