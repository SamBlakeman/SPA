import numpy as np
import glob
from PIL import Image


class MultipleChoice(object):

    def __init__(self, env_params):
        # Set parameters
        self.num_trials = {'training': env_params['num_training_trials'],
                           'test': env_params['num_test_trials']}
        self.num_blocks = {'training': env_params['num_training_blocks'],
                           'test': env_params['num_test_blocks']}
        self.test_interval = env_params['test_interval']
        self.im_height = env_params['image_height']
        self.im_width = env_params['image_width']
        np.random.seed(env_params['random_seed'])

        self.num_categories = len(env_params['object101_categories'])
        self.categories = env_params['object101_categories']

        # Load images
        self.images = {'training': {}, 'test': {}}
        self.num_images = {'training': {}, 'test': {}}
        self.LoadImages()

        # Set up trials
        self.trial_images = {'training': [], 'test': []}
        self.trial_labels = {'training': [], 'test': []}
        self.trial_answers = {'training': [], 'test': []}
        self.SetupTrials()

        # Set variables
        self.trial_nums = {'training': 0, 'test': 0}
        self.block_nums = {'training': 0, 'test': 0}
        self.regime = 'training'
        return

    def LoadImages(self):
        for cat in self.categories:

            img_dir = 'MultipleChoice/Data/101_ObjectCategories/' + cat + '/'
            files = glob.glob(img_dir + '*.jpg')
            num_files = len(files)
            num_training = round((num_files / 100) * 80)

            images = []
            for f in files:
                img = Image.open(f)
                img = img.resize([self.im_height, self.im_width])
                img = np.array(img).astype(np.uint8)
                if (img.shape[-1] == 3):
                    images.append(img)
                else:
                    images.append(np.tile(np.expand_dims(img, axis=-1), (1, 1, 3)))

            self.images['training'][cat] = images[:num_training]
            self.images['test'][cat] = images[num_training:]
            self.num_images['training'][cat] = len(images[:num_training])
            self.num_images['test'][cat] = len(images[num_training:])

    def SetupTrials(self):

        for regime in ['training', 'test']:
            for b in range(self.num_blocks[regime]):
                correct_category = self.categories[np.random.randint(self.num_categories)]

                for t in range(self.num_trials[regime]):
                    positions = np.arange(self.num_categories)
                    np.random.shuffle(positions)
                    images = [[] for _ in range(self.num_categories)]
                    labels = [[] for _ in range(self.num_categories)]

                    for c, cat in enumerate(self.categories):
                        images[positions[c]] = self.images[regime][cat][np.random.randint(self.num_images[regime][cat])]
                        labels[positions[c]] = cat

                    self.trial_images[regime].append(images)
                    self.trial_labels[regime].append(labels)
                    self.trial_answers[regime].append(correct_category)

                # fig, axes = plt.subplots(1, 3)
                #
                # for axis, label, image in zip(axes, labels, images):
                #     axis.imshow(image)
                #     axis.set_title(label)
                #     axis.set_axis_off()
                #
                # fig.suptitle('Block: ' + str(b) + ' Trial: ' + str(t) + ' Answer: ' + str(correct_category))
                # plt.savefig('MultipleChoice/Plots/Block_' + str(b) + 'Trial_' + str(t) + '.jpg')
                # plt.close()


    def Update(self, choice):

        if(self.trial_labels[self.regime][self.trial_nums[self.regime]][choice] == self.trial_answers[self.regime][self.trial_nums[self.regime]]):
            reward = 1
        else:
            reward = 0

        self.trial_nums[self.regime] += 1

        # If block is complete
        if (self.trial_nums[self.regime] % self.num_trials[self.regime] == 0):
            self.block_nums[self.regime] += 1

            if(self.regime == 'training' and self.block_nums['training'] % self.test_interval == 0):
                self.regime = 'test'
            elif(self.regime == 'test' and self.block_nums['test'] % self.num_blocks['test'] == 0):
                self.trial_nums['test'] = 0
                self.block_nums['test'] = 0
                self.regime = 'training'

            # If all training blocks have been done then game is over
            if (self.regime == 'training' and self.block_nums['training'] >= self.num_blocks['training']):
                return reward, True

        return reward, False

    def GetImages(self):
        return self.trial_images[self.regime][self.trial_nums[self.regime]]

    def GetRegime(self):
        return self.regime
