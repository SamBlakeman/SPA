import os
import json
import pickle
import numpy as np
import pandas as pd

def ParseResults(results_dir, directories, parse_particles):
    results_dict = {'dir': [], 'training_results': [], 'test_results': [],
                    'particle_states': [], 'attention': [], 'trial_answers': []}

    for i, dir in enumerate(directories):
        results_dict['dir'].append(dir)
        source_dir = results_dir + dir
        agent_params, env_params = ParseParams(source_dir)

        for key, value in agent_params.items():
            if (key == 'bVerbose' or key == 'bSave_features'):
                pass
            else:
                if (key not in results_dict):
                    results_dict[key] = []
                results_dict[key].append(value)

        for key, value in env_params.items():
            if(key == 'object101_categories'):
                key = 'categories'

            if (key == 'categories'):
                value = ('_').join(value)

            if(key == 'dataset' or key == 'bVerbose' or key == 'imagenet_categories'):
                pass
            else:
                if (key not in results_dict):
                    results_dict[key] = []
                results_dict[key].append(value)

        if parse_particles:
            if (os.path.isfile(source_dir + '/particles.pkl') and
                    agent_params['agent_type'] == 'Particle_Filter' and
                    agent_params['bAttention'] == True):
                with open(source_dir + '/particles.pkl', 'rb') as f:
                    particles = pickle.load(f)
                    results_dict['particle_states'].append(particles['particle_states'])
                    results_dict['attention'].append(particles['attention'])
                with open(source_dir + '/trial_answers.pkl', 'rb') as f:
                    trial_answers = pickle.load(f)
                    results_dict['trial_answers'].append(trial_answers)
            else:
                results_dict['particle_states'].append(None)
                results_dict['attention'].append(None)
                results_dict['trial_answers'].append(None)
        else:
            results_dict['particle_states'].append(None)
            results_dict['attention'].append(None)
            results_dict['trial_answers'].append(None)

        training_results = np.load(source_dir + '/training_results.npy')
        test_results = np.load(source_dir + '/test_results.npy')
        results_dict['training_results'].append(training_results)
        results_dict['test_results'].append(test_results)

    return pd.DataFrame.from_dict(results_dict)


def ParseParams(source_dir):
    with open(source_dir + '/agent_params.txt', 'r') as fp:
        agent_params = json.load(fp)
    with open(source_dir + '/env_params.txt', 'r') as fp:
        env_params = json.load(fp)
    return agent_params, env_params


def ParseFeatureMaps(results_dir, directories):

    all_feature_maps = {}
    for i, dir in enumerate(directories):
        source_dir = results_dir + dir

        if(os.path.isfile(source_dir + '/feature_maps.pkl')):
            with open(source_dir + '/feature_maps.pkl', 'rb') as f:
                feature_maps = pickle.load(f)

            cats = ('_').join(feature_maps.keys())
            if(cats not in all_feature_maps):
                all_feature_maps[cats] = {}

            for key, value in feature_maps.items():
                if(key not in all_feature_maps[cats]):
                    all_feature_maps[cats][key] = []
                all_feature_maps[cats][key] += value

    return all_feature_maps
