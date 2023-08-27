import pygame
from PIL import Image, ImageDraw

from ObjectCollector.Engine.Enums import Shapes

class Object(pygame.sprite.Sprite):


    def __init__(self, level, speed, shape, colour, width, height, x_coord, y_coord):

        super().__init__()

        self.level = level
        self.image = self.DrawColourObject(width, height, colour, shape)

        # Attributes
        self.movement_speed = speed
        self.shape = shape
        self.image.set_colorkey((255, 255, 255))
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.colour = colour
        self.width = width
        self.height = height

        # Position settings
        self.start_xcoord = x_coord
        self.start_ycoord = y_coord
        self.rect.x = x_coord
        self.rect.y = y_coord

        return

    def Update(self):

        self.rect.y += self.movement_speed

        return

    def DrawColourObject(self, width, height, colour, shape):

        surface = pygame.Surface([width, height]).convert()
        surface.fill((255, 255, 255))

        if (shape == Shapes.Rect):
            surface.fill(colour)

        elif (shape == Shapes.Ellipse):
            pygame.draw.ellipse(surface, colour, [0, 0, width, height], 0)

        elif (shape == Shapes.Ring):
            pygame.draw.circle(surface, colour, (int(width / 2), int(height / 2)),
                               int(width / 2), 3)

        elif (shape == Shapes.SemiCircle):

            pil_image = Image.new("RGBA", (width, height))
            pil_draw = ImageDraw.Draw(pil_image)
            pil_draw.pieslice((0, 0, width, 2 * height), 180, 0, fill=colour)
            mode = pil_image.mode
            size = pil_image.size
            data = pil_image.tobytes()
            image = pygame.image.fromstring(data, size, mode)
            image_rect = image.get_rect(center=surface.get_rect().center)
            surface.blit(image, image_rect)

        elif (shape == Shapes.Triangle):
            pygame.draw.polygon(surface, colour, self.Triangle(width, height), 0)

        elif (shape == Shapes.Pentagon):
            pygame.draw.polygon(surface, colour, self.Pentagon(width, height), 0)

        elif (shape == Shapes.Hexagon):
            pygame.draw.polygon(surface, colour, self.Hexagon(width, height), 0)

        elif (shape == Shapes.Octagon):
            pygame.draw.polygon(surface, colour, self.Octagon(width, height), 0)

        elif (shape == Shapes.Parallelogram):
            pygame.draw.polygon(surface, colour, self.Parallelogram(width, height), 0)

        return surface

    def Triangle(self, width, height):
        vertices = [[0, height], [width / 2, 0], [width, height]]
        return vertices

    def Pentagon(self, width, height):
        vertices = [[width / 4, height], [0, (2 * height) / 5], [width / 2, 0], [width, (2 * height) / 5],
                    [(3 * width) / 4, height]]
        return vertices

    def Hexagon(self, width, height):
        vertices = [[width / 2, height], [0, (2 * height) / 3], [0, height / 3], [width / 2, 0], [width, height / 3],
                    [width, (2 * height) / 3]]
        return vertices

    def Octagon(self, width, height):
        vertices = [[width / 3, height], [0, (2 * height) / 3], [0, height / 3], [width / 3, 0], [2 * width / 3, 0],
                    [width, height / 3], [width, (2 * height) / 3], [(2 * width) / 3, height]]
        return vertices

    def Parallelogram(self, width, height):
        vertices = [[0, height], [width / 3, 0], [width, 0], [width - (width / 3), height]]
        return vertices

    def Collide(self, collided_object):



        return