import pygame
import numpy as np

from ObjectCollector.Engine.Object import Object
from ObjectCollector.Engine.PlayerObject import PlayerObject
from ObjectCollector.Engine.Enums import Experiments

class Level(object):

    def __init__(self, env_params, results_dir):

        np.random.seed(env_params['random_seed'])

        self.results_dir = results_dir
        self.experiment = env_params['experiment']
        self.trial_time = env_params['trial_time']
        self.num_trials = env_params['num_trials']
        self.fps = env_params['fps']
        self.screen_width = env_params['screen_width']
        self.screen_height = env_params['screen_height']
        self.shapes = env_params['shapes']
        self.colours = env_params['colours']
        self.size = env_params['size']
        self.speed = env_params['speed']
        self.spawn_speed = env_params['spawn_speed']
        self.trial_frames = self.trial_time * self.fps

        with open(self.results_dir + 'level_rules.txt', 'w') as file:
            file.write("Level Rules:\n")

        self.target_shape = None
        self.target_colour = None
        self.Reset()

        return

    def CreateObject(self):

        if(self.experiment == Experiments.Reward):
            colour = self.colours[np.random.randint(len(self.colours))]
            shape = self.shapes[np.random.randint(len(self.shapes))]

        elif(self.experiment == Experiments.State):
            shape = self.target_shape
            colour = self.target_colour

        x_coord = np.random.randint(self.screen_width - self.size)

        return Object(self, self.speed, shape, colour, self.size, self.size, x_coord, 0)

    def CreatePlayer(self):

        colour = (192, 192, 192)
        shape = 'rect'
        x_coord = (self.screen_width / 2) - (self.size / 2)
        y_coord = self.screen_height - self.size

        return PlayerObject(self, self.speed, shape, colour, self.size * 2, self.size, x_coord, y_coord)

    def Update(self, action):

        self.frame_num += 1

        self.UpdateObjects()
        self.UpdatePlayer(action)
        reward = self.ObjectCollisions()

        if(self.frame_num > self.trial_frames):
            self.trial_num += 1
            self.NewTrial()
            print('     Completed Trial: ' + str(self.trial_num) + '/' + str(self.num_trials))

            if(self.trial_num >= self.num_trials):
                return True, reward, True
            else:
                return True, reward, False

        else:
            return False, reward, False

    def UpdateObjects(self):

        new_object_list = pygame.sprite.Group()
        for object in self.object_list:
            object.Update()
            if (not object.rect.y > self.screen_height):
                new_object_list.add(object)

        if(self.frame_num % self.spawn_speed == 0):
            new_object_list.add(self.CreateObject())

        self.object_list = new_object_list

        return

    def UpdatePlayer(self, action):

        for player in self.player_list:
            player.UpdatePlayer(action)

        return

    def Reset(self):

        # Set new target shape
        new_target = self.shapes[np.random.randint(len(self.shapes))]
        while(self.target_shape == new_target):
            new_target = self.shapes[np.random.randint(len(self.shapes))]

        self.target_shape = new_target
        print('Target Shape:')
        print(self.target_shape)

        with open(self.results_dir + 'level_rules.txt', 'a') as file:
            file.write(str(self.target_shape) + "\n")

        # If changing the state space then also choose new target colour
        if(self.experiment == Experiments.State):
            new_target = self.colours[np.random.randint(len(self.colours))]
            while (self.target_colour == new_target):
                new_target = self.colours[np.random.randint(len(self.colours))]

            self.target_colour = new_target
            print('Target Colour:')
            print(self.target_colour)

            with open(self.results_dir + 'level_rules.txt', 'a') as file:
                file.write(str(self.target_colour) + "\n")


        self.trial_num = 0
        self.NewTrial()

        return

    def NewTrial(self):

        self.frame_num = 0
        self.object_list = pygame.sprite.Group()
        self.player = self.CreatePlayer()
        self.player_list = pygame.sprite.Group()
        self.player_list.add(self.player)

        return

    def ObjectCollisions(self):

        collision_list = pygame.sprite.spritecollide(self.player, self.object_list,
                                                     False, self.ObjectCollide)

        reward = 0

        for object in collision_list:
            object.kill()
            reward += 1

        return reward

    def ObjectCollide(self, player, object):

        # No self collisions
        if (player == object):
            return False
        # Check for collision
        elif (pygame.sprite.collide_circle(player, object)):

            if(self.experiment == Experiments.Reward):
                if(object.shape == self.target_shape):
                    return True
                else:
                    return False

            elif(self.experiment == Experiments.State):
                return True

    def Draw(self, screen):

        screen.fill((0, 0, 0))
        self.object_list.draw(screen)
        self.player_list.draw(screen)

        return