import os

from MultipleChoice.Parsers import ParseResults, ParseFeatureMaps
from MultipleChoice.Plotters import PlotScoresComparison, PlotFeatureMaps, PlotParticleFilters, \
    PlotPairwiseComparison

results_dir = 'MultipleChoice/Results/'
directories = os.listdir(results_dir)
if '.DS_Store' in directories:
    directories.remove('.DS_Store')
results_df = ParseResults(results_dir=results_dir, directories=directories, parse_particles=False)

cats_order = {}
for i, cats in enumerate(results_df['categories'].unique()):
    cats_order[cats] = i

scores_df = results_df
group_list = ['agent_type', 'bAttention'] # which columns to group the data by
PlotScoresComparison(results_df=scores_df, cats_order=cats_order, group_list=group_list)

# SPA vs. no attention
PlotPairwiseComparison(group1=results_df[(results_df['agent_type'] == 'Particle_Filter') &
                           (results_df['bAttention'] == True)],
                       group2=results_df[(results_df['agent_type'] == 'Particle_Filter') &
                           (results_df['bAttention'] == False)],
                       labels=['$SPA$', '$SPA_{ALL}$'])

# SPA vs. self-attention
best_cats, worst_cats = PlotPairwiseComparison(group1=results_df[
    (results_df['agent_type'] == 'Particle_Filter') & (results_df['bAttention'] == True)],
                       group2=results_df[results_df['agent_type'] == 'Self_Attention'],
                       labels=['$SPA$', 'Self-Attention'])

# Compares the best and worst categories
results_df = results_df[results_df['categories'].isin([best_cats, worst_cats])]
results_df = ParseResults(results_dir=results_dir, directories=results_df['dir'].tolist(), parse_particles=True)

cats_order = {}
for i, cats in enumerate(results_df['categories'].unique()):
    cats_order[cats] = i

PlotParticleFilters(results_df=results_df)
PlotFeatureMaps(feature_maps=ParseFeatureMaps(results_dir=results_dir, directories=results_df['dir'].tolist()),
                cats_order=cats_order)
