# Selective particle attention: Rapidly and flexibly selecting features for deep reinforcement learning

This repo contains the code associated with the paper 'Selective particle attention: Rapidly and flexibly selecting features for deep reinforcement learning' (Blakeman and Mareschal, 2022). The paper can be found at the following link: 

https://www.sciencedirect.com/science/article/pii/S0893608022000934

## Dependencies

Please use `python 3.7` for this repo and make sure you have pulled all of the git LFS content. If you are using pip then run the following command in your virtual environment in order to install the required dependencies:

```
pip install -r requirements.txt
```

Otherwise you can manually install the requirements found in ```requirements.txt```

## Multiple Choice Simulations

### Run Simulations

To run an agent on the multiple choice task simply run the ```RunMultipleChoice.py``` file:

```
python RunMultipleChoice.py
```

### Parameters

Parameters for multiple choice simulations are stored in two dictionaries in ```MultipleChoice/Parameters.py```. A brief description of each of the parameters can be found below:

Environment Parameters:

* random_seed --> (int) Used to seed the random generators
* num_repeats --> (int) The number of times you wish to repeat the experiment (each with a different seed)
* image_width --> (int) The width in pixels of the image
* image_height --> (int) The height in pixels of the image
* num_training_trials --> (int) The number of trials during training within a single block (i.e. where the rewarded category does not change)
* num_training_blocks --> (int) The number of blocks during training (i.e. the number of times the rewarded category changes)
* num_test_trials --> (int) The number of trials during testing within a single block
* num_test_blocks --> (int) The number of blocks during testing
* test_interval --> (int) The number of training blocks to perform before testing
* category_permutations --> (int) The number of different image category combinations for the experiment 
* bVerbose --> (bool) Whether to print the result of each trial or not


Agent Parameters:

* agent_type --> (Agents) The type of agent you wish to perform the experiment (Ideal_Observer, Particle_Filter, Self_Attention)
* bAttention --> (bool) Whether to apply attention or not to the input features 
* bVerbose --> (bool) Whether to print key agent quantities or not during the experiment 
* rand_action_prob --> (float) The probability of taking a random action
* bSave_features --> (bool) Whether to save the features of the particle filter
* num_particles --> (int) The number of particles to use in the particle filter
* sigma --> (int) Strength of the top-down attention
* tau --> (int) Strength of the bottom-up attention
* bBottom_up --> (bool) Whether to apply bottom up attention or not (movement step of the particle filter) 
* self_attn_dim --> (int) The dimensionality of the self-attention layer for Self_Attention

### Results

Results are stored in ```MultipleChoice/Results/``` and each simulation is saved in a seperate folder. The folder is named according to the date and time the simulation was ran.

### Plotting

All plots will be saved in ```MultipleChoice/Plots/```. To generate plots (this assumes you have results for all agent types) run ```MultipleChoice/AnalyseResults.py```.

## Object Collector Simulations

### Run Simulations

To run an agent on the object collector 2D game simply run the ```RunObjectCollector.py``` file:

```
python RunObjectCollector.py
```

### Parameters

Parameters for the object collector game are stored in two dictionaries in ```ObjectCollector/Parameters.py```. A brief description of each of the parameters can be found below:

Environment Parameters:

* bHeadless --> (bool) If true then the simulation will run without a GUI
* random_seed --> (int) The number used to seed the random number generators
* num_repeats --> (int) The number of times you wish to repeat the experiment (each with a different seed)
* trial_time --> (int) Duration in seconds that a trial should last for
* num_trials --> (int) The number of trials to perform per level 
* num_levels --> (int) The number of levels to perform (sequentially) 
* spawn_speed --> (int) The number of frames between spawning a new object
* num_actions --> (int) The number of actions available to the agent
* screen_height --> (int) The height in pixels of the screen
* screen_width --> (int) The width in pixels of the screen
* fps --> (int) Frames per second for the simulation
* shapes --> (list(enum)) The different shapes that can be spawned
* colours --> (list(enum)) The different colours that the shapes can be
* size --> (int) The size of the objects
* speed --> (int) The number of pixels per frame that the objects can move
* experiment --> (enum) Whether to change the reward or state structure during the experiment

Agent Parameters:

* agent_type --> (enum) The type of agent you wish to perform the experiment (Particle_Filter, Self_Attention)
* bAttention --> (bool) Whether to apply attention or not (for both agent types)
* bRand_attention --> (bool) Whether to randomly attend to a subset of features or not
* discount_factor --> (float) Disocunt factor to be used in the Bellman equation
* im_width --> (int) The width in pixels of the image to be processed by the agent
* im_height --> (int) The height in pixels of the image to be processed by the agent
* buffer_size --> (int) The length of the buffer to be passed to the LSTM for calculating the n-step return
* save_update --> (int) How often to save the results during training
* frame_skip --> (int) The number of frames between each observation and action of the agent
* num_particles --> (int) The number of particles to use in the particle filter
* tau --> (int) Strength of the bottom-up attention
* sigma --> (int) Strength of the top-down attention
* attention_update --> (int) The number of steps in between updating which features to attend to
* self_attn_dim --> (int) The dimensionality of the self-attention layer for Self_Attention

### Results

Results are stored in ```ObjectCollector/Results/``` and each simulation is saved in a seperate folder. The folder is named according to the date and time the simulation was ran.

### Plotting

All plots will be saved in ```ObjectCollector/Plots/```. To generate plots run ```ObjectCollector/AnalyseResults.py```.

## Authors

* **Sam Blakeman** - *Corresponding Author* - samrobertallan.blakeman@sony.com
* **Denis Mareschal**

## Acknowledgments

* BBSRC for funding the research
* NVIDIA for providing the GPU used to run the simulations
