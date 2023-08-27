import os
import json
import pickle
import pandas as pd

def ParseResults(results_dir):
    results_dict = {'dir': [], 'results': []}

    directories = os.listdir(results_dir)
    if ('.DS_Store' in directories):
        directories.remove('.DS_Store')

    for i, dir in enumerate(directories):
        results_dict['dir'].append(dir)
        source_dir = results_dir + dir
        agent_params, env_params = ParseParams(source_dir)

        for key, value in agent_params.items():
            if (key not in results_dict):
                results_dict[key] = []
            results_dict[key].append(value)

        for key, value in env_params.items():
            if (key not in results_dict):
                results_dict[key] = []
            results_dict[key].append(value)

        with open(source_dir + '/Results.pkl', 'rb') as f:
            results = pickle.load(f)

        results_dict['results'].append(results['rewards'])

    return pd.DataFrame.from_dict(results_dict)


def ParseParams(source_dir):
    with open(source_dir + '/agent_params.txt', 'r') as fp:
        agent_params = json.load(fp)
    with open(source_dir + '/env_params.txt', 'r') as fp:
        env_params = json.load(fp)
    return agent_params, env_params