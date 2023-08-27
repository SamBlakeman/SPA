import pygame
import time
import multiprocessing

from ObjectCollector.Engine.World import World


def Run(env_params, agent_params):
    for random_seed in range(env_params['num_repeats']):
        env_params['random_seed'] = random_seed
        p = multiprocessing.Process(target=RunWorld(agent_params, env_params))
        p.start()
        p.join()

    return


def RunWorld(agent_params, env_params):
    pygame.init()
    bHeadless = env_params['bHeadless']
    screen_width = env_params['screen_width']
    screen_height = env_params['screen_height']
    size = (screen_width, screen_height)
    screen = pygame.display.set_mode(size)
    pygame.display.flip()
    world = World(screen, env_params, agent_params)
    game_over = False
    prev_time = time.time()
    # -------- Main Program Loop -----------
    while not game_over:
        # --- Game logic should go here
        game_over = world.Update()

        # --- Drawing code should go here
        world.Draw()

        # --- Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

    print('Game Over...')
    world.Clear()
    pygame.quit()
