import os
import json
import pickle
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from MultipleChoice.Engine.MultipleChoice import MultipleChoice
from MultipleChoice.Engine.Enums import Agents


class World(object):

    def __init__(self, env_params, agent_params):
        date_time = str(datetime.now())
        date_time = date_time.replace(" ", "_")
        date_time = date_time.replace(".", "_")
        date_time = date_time.replace("-", "_")
        date_time = date_time.replace(":", "_")
        dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        results_dir = dir_path + '/../Results/' + date_time + '/'
        os.mkdir(results_dir)

        with open(results_dir + 'env_params.txt', 'w') as file:
            file.write(json.dumps(env_params))
        with open(results_dir + 'agent_params.txt', 'w') as file:
            file.write(json.dumps(agent_params))

        self.results_dir = results_dir
        self.task = MultipleChoice(env_params)

        if(agent_params['agent_type'] == Agents.Ideal_Observer):
            from MultipleChoice.Agents.Ideal_Observer.Agent import Agent
            self.agent = Agent(results_dir, env_params, agent_params,
                               self.task.trial_labels)

        elif (agent_params['agent_type'] == Agents.Particle_Filter):
            from MultipleChoice.Agents.ParticleFilter.Agent import Agent
            self.agent = Agent(results_dir, env_params, agent_params)

        elif (agent_params['agent_type'] == Agents.Self_Attention):
            from MultipleChoice.Agents.SelfAttention.Agent import Agent
            self.agent = Agent(results_dir, env_params, agent_params)

        self.agent_type = agent_params['agent_type']
        self.bVerbose = env_params['bVerbose']
        self.bSave_features = agent_params['bSave_features']
        self.reward = 0
        self.game_over = False
        self.results = {'training': [], 'test': []}

        self.all_feature_maps = {}
        for cat in self.task.categories:
            self.all_feature_maps[cat] = []

        self.bGet_maps = False

        return

    def Update(self):

        trial_images = self.task.GetImages()
        regime = self.task.GetRegime()

        choice = self.agent.Update(trial_images, self.reward, regime)
        self.reward, self.game_over = self.task.Update(choice)
        self.results[regime].append(self.reward)

        if self.bVerbose:
            print('Trial: ' + str(self.task.trial_nums[regime]) + '/' +
                     str(self.task.num_trials[regime] * self.task.num_blocks[regime]) +
                     ' Choice: ' + str(self.task.trial_labels[regime][self.task.trial_nums[regime]-1][choice]) +
                     ' Reward: ' + str(self.reward))

        if(self.game_over):
            for regime in ['training', 'test']:
                self.PlotResults(regime)
                np.save(self.results_dir + regime + '_results', np.array(self.results[regime]))
            self.results = {}

            if (self.agent_type == Agents.Particle_Filter):
                self.agent.TFGraph.sess.close()

                if (self.agent.bAttention and self.bSave_features):
                    with open(self.results_dir + 'particles.pkl', 'wb') as f:
                        pickle.dump(self.agent.results, f)
                    with open(self.results_dir + 'trial_answers.pkl', 'wb') as f:
                        pickle.dump(self.task.trial_answers['test'], f)
                    with open(self.results_dir + 'feature_maps.pkl', 'wb') as f:
                        pickle.dump(self.all_feature_maps, f)
                self.all_feature_maps = {}
                self.agent.results = {}
                self.task.trial_answers = {}

        else:
            if(self.agent_type == Agents.Particle_Filter and regime == 'training' and self.bGet_maps and self.agent.bAttention and self.bSave_features):
                feature_maps = self.agent.prev_feature_maps
                for i, cat in enumerate(self.task.trial_labels[regime][self.task.trial_nums[regime]-1]):
                    self.all_feature_maps[cat].append(feature_maps[i, :])
            self.bGet_maps = True

        return self.game_over

    def PlotResults(self, regime):
        plt.switch_backend('agg')
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].plot(np.array(self.results[regime]))
        axes[0].vlines(np.arange(self.task.num_blocks[regime]) * self.task.num_trials[regime], 0, 1, 'r')
        axes[1].plot(np.cumsum(self.results[regime]))
        axes[1].plot(np.arange(len(self.results[regime])), np.arange(len(self.results[regime])), 'r')
        axes[1].vlines(np.arange(self.task.num_blocks[regime]) * self.task.num_trials[regime],
                       0, self.task.num_blocks[regime] * self.task.num_trials[regime], 'r')

        fig.suptitle(regime.capitalize() + ' Results')
        plt.savefig(self.results_dir + regime + '_results.png')

        return
