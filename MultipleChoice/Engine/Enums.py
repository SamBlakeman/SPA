from enum import Enum

class Datasets(str, Enum):
    Objects101 = "Objects101"
    ImagenetTiny = "Imagenet_Tiny"

class Agents(str, Enum):
    Ideal_Observer = "Ideal_Observer"
    Particle_Filter = "Particle_Filter"
    Self_Attention = "Self_Attention"
