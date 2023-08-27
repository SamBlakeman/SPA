import numpy as np

from MultipleChoice.Agents.SelfAttention.TFGraph import TFGraph
from keras.applications.vgg16 import preprocess_input

class Agent(object):

    def __init__(self, directory, env_params, agent_params):

        self.directory = directory
        self.bVerbose = agent_params['bVerbose']
        self.rand_action_prob = agent_params['rand_action_prob']
        np.random.seed(env_params['random_seed'])

        self.num_actions = len(env_params['object101_categories'])

        self.choice = 0
        self.prev_value = 0
        self.bStart_learning = False
        self.bRand = False
        self.trial_num = 0

        self.TFGraph = TFGraph(directory=self.directory, vgg16_npy_path='/../../../VGG16/vgg16.npy',
                               input_dim=(env_params['image_height'], env_params['image_width'], 3),
                               rand_action_prob=self.rand_action_prob, num_actions=self.num_actions,
                               attention_dim=agent_params['self_attn_dim'])

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

        if(self.bStart_learning and regime == 'training'):
                loss = self.TFGraph.Train(prev_images=self.prev_images, choice=self.choice, reward=reward)

        self.choice, probs, values, self.bRand = self.TFGraph.SelectAction(input_images=trial_images)
        self.prev_images = trial_images
        self.prev_value = values[self.choice]

        if(self.bVerbose and self.trial_num % self.block_size == 0):
            print('Trial: ' + str(self.trial_num))
            print('    Average reward: ' + str(self.average_reward))
            print('    Values: ' + str(values))
            print('    Loss: ' + str(loss))
            self.n = 0
            self.average_reward = 0

        if(not self.bStart_learning):
            self.bStart_learning = True

        return self.choice
