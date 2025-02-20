import pygame

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165 ,0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

class Spot:
    def __init__(self, row, col, width):
        self.row = row
        self.col = col
        self.width = width
        self.x = col * width
        self.y = row * width
        self.color = BLACK

    def reset(self):
        self.color = BLACK

    def set_color(self, color):
        self.color = color
    
    def is_white(self):
        return self.color == WHITE
    
    def draw(self, win):
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))
    