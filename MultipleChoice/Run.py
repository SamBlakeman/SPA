import glob
import numpy as np

from MultipleChoice.Engine.World import World


def Run(env_params, agent_params):
    category_combinations = GetCategoryCombinations(num_category_permutations=env_params['category_permutations'])

    for random_seed in range(env_params['num_repeats']):
        env_params['random_seed'] = random_seed

        for categories in category_combinations:
            env_params['object101_categories'] = categories
            RunWorld(env_params.copy(), agent_params.copy())
            print('Completed --> Categories: ' + str(categories) + '    Agent: ' + str(agent_params['agent_type']))

        print('COMPLETED RANDOM SEED: ' + str(random_seed) + '/' + str(env_params['num_repeats']))
        print('-------------------------------------------')

    return


def GetCategoryCombinations(num_category_permutations):
    np.random.seed(1)
    categories = glob.glob('MultipleChoice/Data/101_ObjectCategories/*', recursive=False)
    categories = [cat.replace('\\', '/') for cat in categories]
    categories = [folder.split('/')[-1] for folder in categories]
    categories.sort()

    category_combinations = []
    for c in range(num_category_permutations):
        category_combinations.append(np.random.choice(categories, 3, replace=False).tolist())
    return category_combinations


def RunWorld(env_params, agent_params):
    world = World(env_params, agent_params)
    game_over = False

    while not game_over:
        game_over = world.Update()
    print('Game Over...')

    del world
    return
