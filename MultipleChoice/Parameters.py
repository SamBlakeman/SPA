from MultipleChoice.Engine.Enums import Agents

env_params = {'random_seed': 0,
              'num_repeats': 5,
              'image_width': 224,
              'image_height': 224,
              'num_training_trials': 50,
              'num_training_blocks': 200,
              'num_test_trials': 50,
              'num_test_blocks': 10,
              'test_interval': 200,
              'category_permutations': 20,
              'bVerbose': False
              }

agent_params = {'agent_type': Agents.Ideal_Observer,
                'bAttention': False,
                'bVerbose': True,
                'rand_action_prob': .2,
                'bSave_features': True,
                'num_particles': 250,
                'sigma': 10,
                'tau': 10,
                'bBottom_up': True,
                'self_attn_dim': 128}
