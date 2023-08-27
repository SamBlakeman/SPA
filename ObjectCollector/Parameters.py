from ObjectCollector.Engine.Enums import Shapes, Colours, Experiments, Agents


env_params = {'bHeadless': False,
              'random_seed': 0,
              'num_repeats': 5,
              'trial_time': 60,
              'num_trials': 1000,
              'num_levels': 2,
              'spawn_speed': 60, #in frames
              'num_actions': 2,
              'screen_height': 224,
              'screen_width': 224,
              'fps': 60,
              'shapes': [Shapes.Triangle, Shapes.Rect, Shapes.Ellipse, Shapes.Ring],
              'colours': [Colours.Green, Colours.Red, Colours.Blue,
                          Colours.Pink, Colours.Purple, Colours.Yellow, Colours.Orange],
              'size': 15,
              'speed': 1, #pixels per frame
              'experiment': Experiments.Reward
              }

agent_params = {'agent_type': Agents.Particle_Filter,
                'bAttention': True,
                'bRand_attention': False,
                'discount_factor': .99,
                'im_width': 224,
                'im_height': 224,
                'buffer_size': 10,
                'save_update': 10000,
                'frame_skip': 8,
                'num_particles': 250,
                'tau': 1,
                'sigma': 10,
                'attention_update': 1000,
                'self_attn_dim': 128
                }

