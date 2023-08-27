from ObjectCollector.Engine.Object import Object
from ObjectCollector.Parameters import env_params

class PlayerObject(Object):

    def __init__(self, level, speed, shape, colour, width, height, x_coord, y_coord):

        super().__init__(level, speed, shape, colour, width, height, x_coord, y_coord)

        return

    def UpdatePlayer(self, action):

        if(action == 0):
            if(self.rect.x < env_params['screen_width'] - self.width):
                self.rect.x += self.movement_speed
        elif(action == 1):
            if(self.rect.x > 0):
                self.rect.x -= self.movement_speed

        return