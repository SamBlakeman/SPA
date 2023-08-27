import numpy as np


class Agent(object):

    def __init__(self, directory, env_params, agent_params, trial_labels):

        self.trial_labels = trial_labels
        self.directory = directory
        self.rand_action_prob = agent_params['rand_action_prob']
        np.random.seed(env_params['random_seed'])

        self.num_actions = len(env_params['object101_categories'])
        self.action_nums = {'training': 0, 'test': 0}
        self.choice = 0
        self.choice_label = None
        self.bRand = False
        self.regime = 'training'

        return

    def Update(self, trial_images, reward, regime):

        if(regime != self.regime):
            if(regime == 'training'):
                self.action_nums['test'] = 0
            self.regime = regime

        # If rewarded on previous trial then update the label to choose and make choice based on label
        if(reward > 0):
            self.choice_label = self.prev_choice_label
            self.choice = self.trial_labels[self.regime][self.action_nums[self.regime]].index(self.choice_label)
            self.bRand = False
        else:
            # If not rewarded on last choice but random then resume
            if(self.bRand and self.choice_label != None):
                self.choice = self.trial_labels[self.regime][self.action_nums[self.regime]].index(self.choice_label)
                self.bRand = False
            # If not rewarded on last choice and was deliberate then randomly pick another one
            else:
                other_choices = list(range(self.num_actions))
                other_choices.remove(self.choice)
                self.choice = np.random.choice(other_choices)
                self.bRand = False

        # Pick randomly with small probability
        if (np.random.rand() <= self.rand_action_prob):
            self.choice = np.random.randint(self.num_actions)
            self.bRand = True

        self.prev_choice_label = self.trial_labels[self.regime][self.action_nums[self.regime]][self.choice]
        self.action_nums[self.regime] += 1

        return self.choice
