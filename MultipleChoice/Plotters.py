import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm



def PlotFeatureMaps(feature_maps, cats_order):

    mean_feature_maps = {}

    for i, (cats, cats_dict) in enumerate(feature_maps.items()):
        mean_feature_maps[cats] = {}
        for cat, maps in cats_dict.items():
            maps /= np.tile(np.expand_dims(np.sum(maps, axis=1), axis=-1), (1, 512))
            feature_maps[cats][cat] = maps
            mean_feature_maps[cats][cat] = np.mean(maps, axis=0)

    PlotFeatureHeatMaps(mean_feature_maps, cats_order)
    #PlotFeatureHistograms(mean_feature_maps, cats_order, top_num=512)
    #PlotFeatureHistograms(mean_feature_maps, cats_order, top_num=50)
    PlotRawFeatureMaps(feature_maps)

    return


def PlotRawFeatureMaps(feature_maps):
    colours = ['r', 'b', 'g']
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, (cats, cats_dict) in zip(axes, feature_maps.items()):
        b = np.zeros(512)
        for (cat, maps), colour in zip(cats_dict.items(), colours):
            y = np.mean(np.array(maps), axis=0)
            ax.bar(x=np.arange(np.array(maps).shape[1]),
                    height=y, bottom=b,
                    #yerr=np.std(np.array(maps), axis=0),
                    label=cat.replace('_', ' ').capitalize(), alpha=1,
                    color=colour, error_kw=dict(lw=1, capsize=1, capthick=1))
            b += y

        ax.legend(loc='upper right', fontsize='xx-large')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Mean Value', fontsize=20)

    axes[-1].tick_params(axis="x", labelsize=15)
    axes[0].tick_params(axis="y", labelsize=15)
    axes[-1].tick_params(axis="y", labelsize=15)
    axes[-1].set_xlabel('Feature Map Number', fontsize=20)
    axes[-1].set_xlim(0, 512)
    plt.tight_layout()
    plt.savefig('MultipleChoice/Plots/Raw_Feature_Maps.pdf')


def PlotFeatureHistograms(mean_feature_maps, cats_order, top_num):
    fig, axes = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for cats, cats_dict in mean_feature_maps.items():
        diffs = []
        for j, (cat_j, map_j) in enumerate(cats_dict.items()):
            for k, (cat_k, map_k) in enumerate(cats_dict.items()):
                if (j != k):
                    diffs.append(np.abs(map_j - map_k))

        vals = np.reshape(diffs, -1)
        vals.sort()
        vals = vals[-top_num:]
        axes[cats_order[cats]].hist(vals, bins=np.arange(0, .1, .01), color='r')
        axes[cats_order[cats]].set_xlabel('Absolute Differences')
        axes[cats_order[cats]].set_ylabel('Frequency')
        axes[cats_order[cats]].set_title(cats)

    axes[-1].axis('off')
    fig.suptitle('Absolute Differences of VGG16 Feature Maps', fontsize='xx-large')
    fig.savefig('MultipleChoice/Plots/Absolute_Differences_' + str(top_num) + '.png')

    return


def PlotFeatureHeatMaps(mean_feature_maps, cats_order):

    def l2_norm(x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def max_diff(x, y):
        return np.max(np.abs(x - y))

    labels = ['L2 Norms', 'Max Absolute Difference']
    functions = [l2_norm, max_diff]
    for label, function in zip(labels, functions):

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes = axes.reshape(-1)
        subplot_dict = {}
        subplot_values = []

        for cats, cats_dict in mean_feature_maps.items():
            vals = []
            matrix = np.zeros((len(cats_dict), len(cats_dict)))

            for j, (cat_j, map_j) in enumerate(cats_dict.items()):
                for k, (cat_k, map_k) in enumerate(cats_dict.items()):
                    val = function(map_j, map_k)
                    matrix[j, k] = val
                    vals.append(val)

            subplot_dict[cats] = matrix
            subplot_values.append(matrix)
            axes[cats_order[cats]].set_xticks(np.arange(len(cats_dict)))
            axes[cats_order[cats]].set_xticklabels([s.replace('_', ' ').capitalize() for s in list(cats_dict.keys())], fontsize=15)
            axes[cats_order[cats]].set_yticks(np.arange(len(cats_dict)))
            axes[cats_order[cats]].set_yticklabels([s.replace('_', ' ').capitalize() for s in list(cats_dict.keys())], fontsize=15)

        vmin = np.min(subplot_values)
        vmax = np.max(subplot_values)

        for cats, matrix in subplot_dict.items():
            im = axes[cats_order[cats]].imshow(matrix, vmin=vmin, vmax=vmax, cmap='hot')
            axes[cats_order[cats]].set_title('Mean Distance: ' + '{0:.5f}'.format(np.mean(np.reshape(matrix, -1))), fontsize=20)

        fig.subplots_adjust(right=0.8)
        cbar = fig.colorbar(im, cax=fig.add_axes([0.85, 0.15, 0.05, 0.7]))
        cbar.ax.tick_params(labelsize=15)
        fig.savefig('MultipleChoice/Plots/' + label.replace(' ', '_') + '.pdf')

    return


def PlotScoresComparison(results_df, cats_order, group_list):

    num_test_trials = results_df['num_test_trials'].values[0]
    num_test_blocks = results_df['num_test_blocks'].values[0]
    # Plots based on number of training trials
    for training_trials_val, training_trials_group in results_df.groupby('num_training_trials'):
        legend_dict = {}

        # Group by the image categories
        for categories_val, categories_group in training_trials_group.groupby('categories'):
            colours = cm.rainbow(np.linspace(0, 1, categories_group.groupby(group_list).ngroups))

            training_fig, training_ax = plt.subplots(1, 1)
            test_fig, test_ax = plt.subplots(1, 1)

            # Group by agent type
            for colour, (agent_val, agent_group) in zip(colours, categories_group.groupby(group_list)):

                label = ('_').join([str(v) for v in agent_val])

                if('Ideal_Observer' in label):
                    linestyle = '--'
                    linewidth = 3
                    colour = 'b'
                    label = 'Ideal Observer'
                elif('Particle_Filter_False' in label):
                    linestyle = ':'
                    linewidth = 3
                    colour = 'm'
                    label = '$SPA_{ALL}$'
                elif('Particle_Filter_True' in label):
                    linestyle = '-'
                    linewidth = 3
                    colour = 'r'
                    label = '$SPA$'
                elif('Self_Attention' in label):
                    linestyle = '-'
                    linewidth = 3
                    colour = 'y'
                    label = 'Self-Attention'

                legend_dict[label] = (linestyle, colour)

                PlotTrainingResults(results=np.array(agent_group['training_results'].tolist()), ax=training_ax,
                                    colour=colour, linestyle=linestyle, linewidth=linewidth)

                PlotTestResults(num_test_trials=num_test_trials, num_test_blocks=num_test_blocks,
                                results=np.array(agent_group['test_results'].tolist()), ax=test_ax,
                                colour=colour, linestyle=linestyle, linewidth=linewidth)


            for label, vals in legend_dict.items():
                training_ax.plot(.5, .5, label=label, color=vals[1], linestyle=vals[0], linewidth=3)
            training_ax.set_xlabel('Trial', fontsize='x-large')
            training_ax.set_ylabel('Score', fontsize='x-large')
            training_ax.spines['top'].set_visible(False)
            training_ax.spines['right'].set_visible(False)
            training_ax.legend(fontsize='x-large')
            training_fig.tight_layout()
            training_fig.savefig('MultipleChoice/Plots/TrainingComparison' + '_' + str(categories_val) + '_' + str(training_trials_val) + 'TrainingTrials.pdf')

            for label, vals in legend_dict.items():
                test_ax.plot(.5, .5, label=label, color=vals[1], linestyle=vals[0], linewidth=3)
            test_ax.set_xlabel('Trial', fontsize='x-large')
            test_ax.set_ylabel('Score', fontsize='x-large')
            test_ax.spines['top'].set_visible(False)
            test_ax.spines['right'].set_visible(False)
            test_ax.legend(fontsize='x-large')
            test_fig.tight_layout()
            test_fig.savefig('MultipleChoice/Plots/TestComparison' + '_' + str(categories_val) + '_' + str(training_trials_val) + 'TrainingTrials.pdf')

    return


def PlotPairwiseComparison(group1, group2, labels):
    """Returns the best and worst categories during training for group 1"""

    training_y = []
    test_y = []
    cats = []

    for categories, categories_group in group1.groupby('categories'):
        training_y.append(np.mean(np.sum(categories_group['training_results'].tolist(), axis=1), axis=0))
        test_y.append(np.mean(np.sum(categories_group['test_results'].tolist(), axis=1), axis=0))
        cats.append(categories)

    best_cats = cats[np.argmax(training_y)]
    print(labels[0] + ' Scores:')
    print('Max training score: ' + str(np.amax(training_y)))
    print('Categories: ' + str(best_cats))

    worst_cats = cats[np.argmin(training_y)]
    print('Min training score: ' + str(np.amin(training_y)))
    print('Categories: ' + str(worst_cats) + '\n')

    training_x = []
    test_x = []

    for categories, categories_group in group2.groupby('categories'):
        training_x.append(np.mean(np.sum(categories_group['training_results'].tolist(), axis=1), axis=0))
        test_x.append(np.mean(np.sum(categories_group['test_results'].tolist(), axis=1), axis=0))

    training_x = np.array(training_x)
    training_y = np.array(training_y)

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    training_fig, training_axes = plt.subplots(1, 1)
    test_fig, test_axes = plt.subplots(1, 1)
    training_axes.ticklabel_format(style='sci', axis='both', scilimits=(0, 0))

    colors = ['r', 'b']
    training_axes.scatter(training_x, training_y, color=[colors[i] for i in training_y > training_x])
    test_axes.scatter(test_x, test_y, color=[colors[i] for i in test_y > test_x])

    min_val = np.min(np.concatenate([training_x, training_y]))
    max_val = np.max(np.concatenate([training_x, training_y]))
    training_axes.plot([min_val, max_val], [min_val, max_val], 'k-')
    training_axes.axis('equal')
    training_axes.set_aspect('equal', 'box')

    min_val = np.min(np.concatenate([test_x, test_y]))
    max_val = np.max(np.concatenate([test_x, test_y]))
    test_axes.plot([min_val, max_val], [min_val, max_val], 'k-')
    test_axes.axis('equal')
    test_axes.set_aspect('equal', 'box')

    training_axes.set_ylabel(labels[0], fontsize='x-large')
    training_axes.set_xlabel(labels[1], fontsize='x-large')

    test_axes.set_ylabel(labels[0], fontsize='x-large')
    test_axes.set_xlabel(labels[1], fontsize='x-large')

    training_axes.spines['top'].set_visible(False)
    training_axes.spines['right'].set_visible(False)
    test_axes.spines['top'].set_visible(False)
    test_axes.spines['right'].set_visible(False)

    training_fig.tight_layout()
    training_fig.savefig('MultipleChoice/Plots/PairwiseComparisonPlotTraining_' + str.join('_', labels) + '.pdf')
    plt.close(training_fig)

    test_fig.tight_layout()
    test_fig.savefig('MultipleChoice/Plots/PairwiseComparisonPlotTest_' + str.join('_', labels) + '.pdf')
    plt.close(test_fig)

    return best_cats, worst_cats


def PlotTrainingResults(results, ax, colour, linestyle, linewidth):
    values = np.cumsum(results, axis=1)
    x = np.arange(values.shape[1])
    y = np.mean(values, axis=0)
    error = np.std(values, axis=0)
    ax.plot(x, y, color=colour, linestyle=linestyle, linewidth=linewidth)
    ax.fill_between(x, y - error, y + error, color=colour, alpha=.5)
    return


def PlotTestResults(num_test_trials, num_test_blocks, results, ax, colour, linestyle, linewidth):

    num_tests = int(results.shape[1] / (num_test_trials * num_test_blocks))
    results = np.reshape(results, (results.shape[0], num_tests, num_test_trials * num_test_blocks))

    x_start = 0
    for test in range(num_tests):
        values = np.cumsum(results[:, test, :], axis=1)
        x = np.arange(x_start, values.shape[1] + x_start)
        y = np.mean(values, axis=0)
        error = np.std(values, axis=0)
        if(x_start == 0):
            ax.plot(x, y, color=colour, linestyle=linestyle, linewidth=linewidth)
        else:
            ax.plot(x, y, color=colour, linestyle=linestyle, linewidth=linewidth)
        ax.fill_between(x, y - error, y + error, color=colour, alpha=.5)
        ax.vlines(x=np.linspace(0, num_test_trials * num_test_blocks, num_test_blocks + 1) + x_start,
                  ymin=0, ymax=num_test_trials * num_test_blocks, linewidth=.5, linestyle=':')
        x_start = x[-1] + 1
    return


def PlotParticleFilters(results_df):
    select_df = results_df[(results_df['agent_type'] == 'Particle_Filter') &
                           (results_df['bAttention'] == True)]

    for particle_states, bAttention, attention, tau, num_particles, \
        sigma, num_test_trials, num_test_blocks, \
        trial_answers, cats in zip(select_df['particle_states'].tolist(),
                             select_df['bAttention'].tolist(),
                             select_df['attention'].tolist(),
                             select_df['tau'].tolist(),
                             select_df['num_particles'].tolist(),
                             select_df['sigma'].tolist(),
                             select_df['num_test_trials'].tolist(),
                             select_df['num_test_blocks'].tolist(),
                             select_df['trial_answers'].tolist(),
                             select_df['categories'].tolist()):

        fig, ax = plt.subplots(1, 1)
        plt.imshow(np.array(attention), cmap='hot')
        plt.hlines(y=np.arange(0, num_test_trials * num_test_blocks, num_test_trials), xmin=0, xmax=512, color='b', linestyle='-')
        for i in np.arange(0, num_test_trials * num_test_blocks, num_test_trials):
            plt.text(np.array(attention).shape[1] + 20, i + (num_test_trials / 2), trial_answers[i].replace('_', ' ').capitalize())
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        plt.ylabel('Trial Number', fontsize='x-large')
        plt.xlabel('Feature Number', fontsize='x-large')
        plt.tight_layout()
        plt.savefig('MultipleChoice/Plots/' + str(cats) + '_' +
                    str(tau) + '_' +
                    str(num_particles) + '_' +
                    str(sigma) + '_' +
                    '_attention.pdf')
    return
