import mnist_loader as mnist_loader
from network import Network
from knn import KNN
import pygame
from spot import Spot, GREY, BLACK, WHITE, GREEN, RED
from random import randint
import numpy as np

WIDTH = 784
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("Handwritten digits detector")
pygame.font.init()
FONT = pygame.font.Font(None, 50)
SMALL_FONT = pygame.font.Font(None, 30)

def prepare_data_for_knn(data):
    """Convert MNIST data format to KNN format."""
    knn_data = []
    for x, y in data:
        if isinstance(y, np.ndarray) and y.shape == (10, 1):
            label = np.argmax(y)
        else:
            label = y
        knn_data.append((x, label))
    return knn_data

def train_network():
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    
    net = Network([784, 30, 10])
    
    if not net.is_cached():
        print("Training neural network...")
        net.Stochastic_Gradient_Descent_Training(training_data, 30, 10, 3.0, test_data=test_data)
    
    print("\nEvaluating neural network on test data...")
    net.print_metrics_report(test_data)
    
    return net

def train_knn():
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    
    # Convert to KNN format
    training_data = prepare_data_for_knn(list(training_data))
    test_data = prepare_data_for_knn(list(test_data))
    
    knn = KNN(k=5, num_bins=16)
    
    model_path = "../cache/knn_model_k5.pkl"
    if knn.load_model(model_path):
        print("Using cached KNN model")
    else:
        print("Training KNN classifier...")
        knn.fit(training_data, use_spatial=True)
        knn.save_model(model_path)
    
    print("\nEvaluating KNN on test data (1000 samples)...")
    knn.print_metrics_report(test_data[:1000], use_spatial=True, distance_metric='chi_square')
    
    return knn

def grid_to_input(grid, rows):
    """Convert a grid to a neural network input (784x1 array)."""
    input_vector = np.zeros((rows * rows, 1))
    for row in range(rows):
        for col in range(rows):
            spot = grid[row][col]
            intensity = spot.color[0] / 255.0
            input_vector[row * rows + col] = intensity
    return input_vector

def make_grid(rows, width):
    grid = []
    gap = width // rows
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

def draw(win, grid, rows, width, prediction_nn=None, prediction_knn=None, use_knn=False):
    win.fill(BLACK)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)
    
    # Draw predictions
    y_offset = 20
    if use_knn and prediction_knn is not None:
        text_surface = FONT.render(f"KNN: {prediction_knn}", True, GREEN)
        win.blit(text_surface, (width // 2 - 80, y_offset))
    elif prediction_nn is not None:
        text_surface = FONT.render(f"NN: {prediction_nn}", True, WHITE)
        win.blit(text_surface, (width // 2 - 80, y_offset))
    
    # Draw instructions
    instructions = [
        "Left Click: Draw",
        "Right Click: Erase",
        "N: Predict (NN)",
        "K: Predict (KNN)",
        "C: Clear",
        "Q: Quit"
    ]
    
    y_pos = width - 200
    for instruction in instructions:
        text = SMALL_FONT.render(instruction, True, GREY)
        win.blit(text, (10, y_pos))
        y_pos += 30
    
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
    
    net = None
    knn = None
    net = train_network()
    knn = train_knn()
    
    if net is None and knn is None:
        print("No classifier loaded. Exiting.")
        return
    
    run = True
    predicted_digit_nn = None
    predicted_digit_knn = None
    current_mode = 'nn' if net is not None else 'knn'
    
    while run:
        use_knn = (current_mode == 'knn')
        draw(win, grid, ROWS, width, predicted_digit_nn, predicted_digit_knn, use_knn)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT - Draw
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if 0 <= row <= 27 and 0 <= col <= 27:
                    spot = grid[row][col]
                    spot.set_color(WHITE)
                    
                    # Add gradient effect to neighbors
                    neighbors = [
                        (row, col - 1),
                        (row - 1, col),
                        (row, col + 1),
                        (row + 1, col)
                    ]
                    
                    for nr, nc in neighbors:
                        if 0 <= nr <= 27 and 0 <= nc <= 27:
                            neighbour = grid[nr][nc]
                            if not neighbour.is_white():
                                rand = randint(0, 160)
                                neighbour.set_color((255 - rand, 255 - rand, 255 - rand))

            elif pygame.mouse.get_pressed()[2]:  # RIGHT - Erase
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if 0 <= row <= 27 and 0 <= col <= 27:
                    spot = grid[row][col]
                    spot.reset()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Clear
                    grid = make_grid(ROWS, width)
                    predicted_digit_nn = None
                    predicted_digit_knn = None

                if event.key == pygame.K_n and net is not None:  # Predict with Neural Network
                    input_vec = grid_to_input(grid, ROWS)
                    output = net.feed_forward(input_vec)
                    argmax = np.argmax(output)
                    if output[argmax] > 0.5 and np.sum(output) - output[argmax] < 0.1:
                        predicted_digit_nn = str(argmax)
                    else:
                        predicted_digit_nn = "Inconclusive"
                    current_mode = 'nn'
                    predicted_digit_knn = None

                if event.key == pygame.K_k and knn is not None:  # Predict with KNN
                    print("Predicting with knn")
                    input_vec = grid_to_input(grid, ROWS)
                    prediction = knn.predict(input_vec, use_spatial=True, distance_metric='chi_square')
                    predicted_digit_knn = str(prediction)
                    current_mode = 'knn'
                    predicted_digit_nn = None
                
                if event.key == pygame.K_q:  # Quit
                    run = False

    pygame.quit()

if __name__ == "__main__":
    main(WIN, WIDTH)