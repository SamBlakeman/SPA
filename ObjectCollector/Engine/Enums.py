from enum import Enum

class Colours():
    Black = (0, 0, 0)
    White = (255, 255, 255)
    Green = (0, 255, 0)
    Red = (255, 0, 0)
    Blue = (0, 0, 255)
    LightBlue = (0, 255, 255)
    Orange = (255, 140, 0)
    Pink = (255, 51, 255)
    Yellow = (255, 255, 0)
    Purple = (127, 0, 255)
    Brown = (165, 42, 42)
    DarkGrey = (96, 96, 96)
    LightGrey = (192, 192, 192)

class Shapes(str, Enum):
    Rect = 'rect'
    Ellipse = 'ellipse'
    Ring = 'ring'
    SemiCircle = 'semicircle'
    Triangle = 'triangle'
    Pentagon = 'pentagon'
    Hexagon = 'hexagon'
    Octagon = 'octagon'
    Parallelogram = 'parallelogram'

class Experiments(str, Enum):
    Reward = 'reward'
    State = 'state'

class Agents(str, Enum):
    Particle_Filter = "Particle_Filter"
    Self_Attention = "Self_Attention"
