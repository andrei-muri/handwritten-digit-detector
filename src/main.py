import mnist_loader as mnist_loader
from network import Network
from knn import KNN
from bayesian import Bayesian
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
def prepare_data(data):
    output_data = []
    for x, y in data:
        if isinstance(y, np.ndarray) and y.shape == (10, 1):
            label = np.argmax(y)
        else:
            label = y
        output_data.append((x, label))
    return output_data

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
    training_data = prepare_data(list(training_data))
    test_data = prepare_data(list(test_data))
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

def train_bayesian():
    training_data, _, test_data = mnist_loader.load_data_wrapper()
    training_data = prepare_data(list(training_data))
    test_data = prepare_data(list(test_data))

    bayes = Bayesian()
    model_path = "../cache/bayesian_model.pkl"
    if bayes.load_model(model_path):
        print("Loaded cached Bayesian model")
    else:
        print("Training Bayesian classifier...")
        bayes.fit(training_data)
        bayes.save_model(model_path)
    print("\nEvaluating Bayesian on test data (1000 samples)...")
    bayes.print_metrics_report(test_data[:1000])
    return bayes

def grid_to_input(grid, rows):
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
    for i in range(rows):
        pygame.draw.line(win, GREY, (i * gap, 0), (i * gap, width))
    for j in range(rows):
        pygame.draw.line(win, GREY, (0, j * gap), (width, j * gap))

def draw(win, grid, rows, width, prediction_nn=None, prediction_knn=None, prediction_bayes=None, mode="nn"):
    win.fill(BLACK)
    for row in grid:
        for spot in row:
            spot.draw(win)
    draw_grid(win, rows, width)

   
    if mode == "bayes" and prediction_bayes is not None:
        text = FONT.render(f"Bayes: {prediction_bayes}", True, GREEN)
    elif mode == "knn" and prediction_knn is not None:
        text = FONT.render(f"KNN: {prediction_knn}", True, GREEN)
    elif prediction_nn is not None:
        text = FONT.render(f"NN: {prediction_nn}", True, WHITE)
    else:
        text = FONT.render("Draw!", True, GREY)
    win.blit(text, (width//2 - 100, 20))

    inst = ["LMB:Draw", "RMB:Erase", "N=NN", "K=KNN", "B=Bayes", "C=Clear", "Q=Quit"]
    for i, s in enumerate(inst):
        t = SMALL_FONT.render(s, True, GREY)
        win.blit(t, (20, width - 180 + i*30))

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
    knn = train_knn()
    bayes = train_bayesian()

    run = True
    pred_nn = pred_knn = pred_bayes = None
    mode = "nn"

    while run:
        draw(win, grid, ROWS, width, pred_nn, pred_knn, pred_bayes, mode)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if 0 <= row < 28 and 0 <= col < 28:
                    grid[row][col].set_color(WHITE)
                    for dr, dc in [(0,-1),(0,1),(-1,0), (1,0)]:
                        r, c = row+dr, col+dc
                        if 0 <= r < 28 and 0 <= c < 28 and not grid[r][c].is_white():
                            rand = randint(0,160)
                            grid[r][c].set_color((255-rand,255-rand,255-rand))

            if pygame.mouse.get_pressed()[2]:
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                if 0 <= row < 28 and 0 <= col < 28:
                    grid[row][col].reset()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    grid = make_grid(ROWS, width)
                    pred_nn = pred_knn = pred_bayes = None

                elif event.key == pygame.K_n and net:
                    vec = grid_to_input(grid, ROWS)
                    out = net.feed_forward(vec)
                    pred = np.argmax(out)
                    pred_nn = str(pred)
                    pred_knn = pred_bayes = None
                    mode = "nn"

                elif event.key == pygame.K_k and knn:
                    vec = grid_to_input(grid, ROWS)
                    pred_knn = str(knn.predict(vec, use_spatial=True, distance_metric='chi_square'))
                    pred_nn = pred_bayes = None
                    mode = "knn"

                elif event.key == pygame.K_b and bayes:
                    vec = grid_to_input(grid, ROWS)
                    pred_bayes = str(bayes.predict(vec))
                    pred_nn = pred_knn = None
                    mode = "bayes"

                elif event.key == pygame.K_q:
                    run = False

    pygame.quit()

if __name__ == "__main__":
    main(WIN, WIDTH)