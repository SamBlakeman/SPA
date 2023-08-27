import os

from ObjectCollector.Run import Run
from ObjectCollector.Parameters import env_params, agent_params

if(env_params['bHeadless']):
    os.environ['SDL_VIDEODRIVER'] = 'dummy'

Run(env_params, agent_params)
