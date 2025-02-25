import mnist_loader as mnist_loader
from network import Network
import pygame
from spot import Spot, GREY, BLACK, WHITE
from random import randint
import numpy as np


WIDTH = 784

WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Handwritten digits detector")
pygame.font.init()
FONT = pygame.font.Font(None, 50)

def train_network():
	training_data, _, _ = mnist_loader.load_data_wrapper()
	training_data = list(training_data)
	net = Network([784, 30, 10])

	if not net.is_cached():
		net.Stochastic_Gradient_Descent_Training(training_data, 30, 10, 3.0)
	return net

def grid_to_input(grid, rows):
    """Convert a grid to a neural network input (784x1 array).
    
    - Black (empty spots) → 0
    - White (drawn spots) → 1
    - Other shades are normalized to [0,1] range.
    """
    input_vector = np.zeros((rows * rows, 1))  # 784x1 vector
    
    for row in range(rows):
        for col in range(rows):
            spot = grid[row][col]
            # Normalize color intensity (R, G, B are equal, so we take one channel)
            intensity = spot.color[0] / 255.0  # Normalize between 0 and 1
            input_vector[row * rows + col] = intensity  # Flattened index

    return input_vector

    
def make_grid(rows, width):
    grid = []
    gap =  width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
             spot = Spot(i, j, gap)
             grid[i].append(spot)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    cols = rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (i * gap, 0), (i * gap, width))
    for j in range(cols):
        pygame.draw.line(win, GREY, (0, j * gap), (width, j * gap))

def draw(win, grid, rows, width, prediction=None):
	win.fill(BLACK)

	for row in grid:
		for spot in row:
			spot.draw(win)

	draw_grid(win, rows, width)

	if prediction is not None:
		text_surface = FONT.render(f"Digit: {prediction}", True, WHITE)
		win.blit(text_surface, (width // 2 - 50, 20))
	pygame.display.update()
    
def get_clicked_pos(pos, rows, width):
	gap = width // rows
	y, x = pos

	row = x // gap
	col = y // gap

	return row, col


def main(win, width):
	
	ROWS = 28
	grid = make_grid(ROWS, width)
	net = train_network()
	run = True
	predicted_digit = None
	while run:
		draw(win, grid, ROWS, width, predicted_digit)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False

			if pygame.mouse.get_pressed()[0]: # LEFT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				if row < 0 or row > 27 or col < 0 or col > 27:
					continue 
				spot = grid[row][col]
				spot.set_color(WHITE)
				if col > 0:
					neighbour = grid[row][col - 1]
					if(not neighbour.is_white()):
						rand = randint(0, 160)
						neighbour.set_color((255 - rand, 255 - rand, 255 - rand))
						
				if row > 0:
					neighbour = grid[row - 1][col]
					if(not neighbour.is_white()):
						rand = randint(0, 160)
						neighbour.set_color((255 - rand, 255 - rand, 255 - rand))
						
				if col < 27:
					neighbour = grid[row][col + 1]
					if(not neighbour.is_white()):
						rand = randint(0, 160)
						neighbour.set_color((255 - rand, 255 - rand, 255 - rand))
						
				if row < 27:
					neighbour = grid[row + 1][col]
					if(not neighbour.is_white()):
						neighbour.set_color(WHITE)
						
				

			elif pygame.mouse.get_pressed()[2]: # RIGHT
				pos = pygame.mouse.get_pos()
				row, col = get_clicked_pos(pos, ROWS, width)
				if row < 0 or row > 27 or col < 0 or col > 27:
					continue
				spot = grid[row][col]
				spot.reset()
			

			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_c:
					grid = make_grid(ROWS, width)
					predicted_digit = None

				if event.key == pygame.K_n:
					input = grid_to_input(grid, ROWS)
					output = net.feed_forward(input)
					argmax = np.argmax(output)
					if(output[argmax] > 0.9 and np.sum(output) - output[argmax] < 0.1):
						predicted_digit = str(argmax)
					else:
						predicted_digit = "Inconclussive"
				
                
				

	pygame.quit()


main(WIN, WIDTH)
