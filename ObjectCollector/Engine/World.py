import os
import json
import pygame
import numpy as np
from datetime import datetime
from PIL import Image

from ObjectCollector.Engine.Level import Level
from ObjectCollector.Engine.Enums import Agents


class World(object):

    def __init__(self, screen, env_params, agent_params):


        self.num_levels = env_params['num_levels']
        self.num_actions = env_params['num_actions']
        self.screen = screen
        self.level_num = 0
        self.bTrial_over = False
        self.bLevel_over = False
        self.bGame_over = False
        self.action = 0
        self.reward = 0
        self.results_dir = self.MakeResultsDirectory()

        self.level = Level(env_params, self.results_dir)

        if (agent_params['agent_type'] == Agents.Particle_Filter):
            from ObjectCollector.Agents.ParticleFilter.Agent import Agent
            self.model = Agent(self.results_dir, env_params, agent_params)

        elif (agent_params['agent_type'] == Agents.Self_Attention):
            from ObjectCollector.Agents.SelfAttention.Agent import Agent
            self.model = Agent(self.results_dir, env_params, agent_params)

        with open(self.results_dir + 'env_params.txt', 'w') as file:
            file.write(json.dumps(env_params))
        with open(self.results_dir + 'agent_params.txt', 'w') as file:
            file.write(json.dumps(agent_params))

        #self.ti = 0

        return

    def MakeResultsDirectory(self):

        date_time = str(datetime.now())
        date_time = date_time.replace(" ", "_")
        date_time = date_time.replace(".", "_")
        date_time = date_time.replace("-", "_")
        date_time = date_time.replace(":", "_")
        dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        results_dir = dir_path + '/../Results/' + date_time + '/'
        os.mkdir(results_dir)

        return results_dir


    def Update(self):

        #pxarray = np.array(pygame.surfarray.array3d(self.screen))
        #im = Image.fromarray(pxarray)
        #im.save('ObjectCollector/Plots/' + str(self.ti) + ".jpeg")
        #self.ti += 1

        pxarray = np.array(pygame.surfarray.array3d(self.screen))
        self.action = self.model.Update(self.reward, pxarray, self.bTrial_over)
        self.bTrial_over, self.reward, self.bLevel_over = self.level.Update(self.action)

        if (self.bLevel_over):
            self.level_num += 1
            print('Completed Level: ' + str(self.level_num) + '/' + str(self.num_levels))

            if (self.level_num >= self.num_levels):
                self.bGame_over = True
            else:
                self.level.Reset()

        return self.bGame_over

    def Draw(self):

        self.level.Draw(self.screen)

        return

    def Clear(self):

        self.model.Clear()

        return
