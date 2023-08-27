import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from ObjectCollector.Parsers import ParseResults



results_df = ParseResults(results_dir="Results/State/")
group_list = ['agent_type', 'bAttention', 'bRand_attention']

bin_sizes = [20]
results_dict = {}

for agent_val, agent_group in results_df.groupby(group_list):
    label = ('_').join([str(v) for v in agent_val])
    if ('Particle_Filter_False_False' in label):
        label = '$SPA_{ALL}$'
    elif ('Particle_Filter_True_True' in label):
        label = '$SPA_{RANDOM}$'
    elif ('Particle_Filter_True_False' in label):
        label = '$SPA$'
    elif ('Self_Attention' in label):
        label = 'Self-Attention'

    results_dict[label] = np.array(agent_group['results'].tolist())


for bin in bin_sizes:

    fig, ax = plt.subplots(1, 1)
    all_fig, all_ax = plt.subplots(1, 1)
    max_val = 0

    for i, (label, results) in enumerate(results_dict.items()):

        if ('ALL' in label):
            linestyle = '--'
            linewidth = 2
            colour = 'b'
        elif ('RANDOM' in label):
            linestyle = ':'
            linewidth = 2
            colour = 'm'
        elif ('Self' in label):
            linestyle = '-'
            linewidth = 2
            colour = 'y'
        else:
            linestyle = '-'
            linewidth = 2
            colour = 'r'

        # single level
        rem = 1000 % bin
        r = np.copy(results[:, :1000-rem])
        r = np.reshape(r, (results.shape[0], -1, bin))
        r = np.mean(r, axis=-1)

        ax.plot(np.arange(0, 1000 - rem, bin) + bin, np.mean(r, axis=0),
                color=colour, linestyle=linestyle, linewidth=linewidth, label=label)
        ax.fill_between(np.arange(0, 1000 - rem, bin) + bin, np.mean(r, axis=0) - np.std(r, axis=0),
                        np.mean(r, axis=0) + np.std(r, axis=0), color=colour, alpha=.25)

        # all levels
        rem = results.shape[1] % bin
        if(rem != 0):
            r = np.copy(results[:, :-rem])
        else:
            r = np.copy(results)

        r = np.reshape(r, (results.shape[0], -1, bin))
        r = np.mean(r, axis=-1)

        all_ax.plot(np.arange(0, results.shape[1] - rem, bin) + bin, np.mean(r, axis=0),
                    color=colour, linestyle=linestyle, linewidth=linewidth, label=label)
        all_ax.fill_between(np.arange(0, results.shape[1] - rem, bin) + bin, np.mean(r, axis=0) - np.std(r, axis=0),
                            np.mean(r, axis=0) + np.std(r, axis=0), color=colour, alpha=.25)

        if(np.amax(np.amax(r)) > max_val):
            max_val = np.amax(np.amax(r))

        if(i == 0):
            min_val = np.amin(np.amin(r))
        elif(np.amin(np.amin(r)) < min_val):
            min_val = np.amin(np.amin(r))

    ax.legend()
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig('Plots/ResultsComparison.pdf')

    all_ax.vlines(1000, min_val, max_val, color='k', linestyle='--')
    all_ax.legend()
    all_ax.set_ylabel('Score')
    all_ax.set_xlabel('Episode')
    all_ax.spines['top'].set_visible(False)
    all_ax.spines['right'].set_visible(False)
    all_fig.savefig('Plots/ResultsComparisonAllLevels.pdf')



