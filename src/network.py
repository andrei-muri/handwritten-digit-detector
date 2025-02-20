# concepts and code inspired by Michael Nielsen @ http://neuralnetworksanddeeplearning.com/index.html

import numpy as np
from random import shuffle
import os
import pickle

class Network:
    def __init__(self, sizes, load_from_file=True):
        """``sizes`` is a list that contains the numbers of neurons in each 
        layer. ``num_layers`` represents the number of layers. ``biases`` and ``weights`` are
        a list of lists and matrices, representing the combination of weights and biases at each layer.
        so ``self.weights[0]`` represents the matrix of weights that translates the input into the second layer 
        (without the sigmoid part)."""
        self.sizes = sizes
        self.num_layers = len(sizes)
        if load_from_file and os.path.exists("..\\cache\\network_params.pkl"):
            print("Loading saved network parameters...")
            self.load_parameters()
            self.cached = True
        else:
            print("No saved parameters found. Initializing randomly...")
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
            self.cached = False

    def is_cached(self):
        return self.cached

    def feed_forward(self, a):
        """Actual computation of the output based on the weights and biases. ``a`` is the input"""
        for i in range(self.num_layers - 1):
            a = self.sigmoid(np.dot(self.weights[i], a) + self.biases[i])
        return a
    
    def Stochastic_Gradient_Descent_Training(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """``training data`` is a list of tuples (x, y) ``x`` being the neuronal input and ``y`` the desired output"""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for i in range(epochs):
            shuffle(training_data)
            # create the mini batches
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # now for each batch compute the gradient and subtract from the input space
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {i}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {i} complete")
        self.save_parameters()
        print("Training complete. Parameters saved.")

    
    def update_mini_batch(self, mini_batch, learning_rate):
        """This method updates the weights and baises computing the gradient for the mini_batch
        and subtracting it from the current weights and biases"""
        # accumulators for the gradients:
        nabla_weights = [np.zeros(w.shape) for w in self.weights]
        nabla_biases = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            # compute the gradient for each input
            delta_nabla_weights, delta_nabla_biases = self.backpropagation(x, y)
            # add to existing accumulator
            nabla_weights = [nw + nw_x for nw, nw_x in zip(nabla_weights, delta_nabla_weights)]
            nabla_biases = [nb + nb_x for nb, nb_x in zip(nabla_biases, delta_nabla_biases)]
        # subtract the median from the current weights and biases
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_weights)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_biases)]

    
    def backpropagation(self, x, y):
        """We start by computing the error (just partial derivative of cost w.r.t. to the Z^l) for the output
        then we propagte backwards the computing of the gradient with formulas that can be found on the internet
        so I will not state them here. The main idea is that we want to compute the partial derivative of the cost
        w.r.t. to all the weights and the biases. We will do this in a vectorial manner."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feed forward the input to obtain the output and intermediary activations
        activation = x
        activations = [x] # list to store all the activations
        zs = [] # list to store all the z vectors
        for b, w in zip(self.biases, self.weights):
            #print(f"{w.shape}, {activation.shape}, {b.shape}")
            z = (w @ activation) + b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        
        # First compute delta (error) for the output 
        delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_w, nabla_b)


    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


    def cost_derivative(self, output, y):
        """``y`` is the expected output. ``output`` is the actual output"""
        return (output - y)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def save_parameters(self):
        """Save weights and biases to a file."""
        with open("..\\cache\\network_params.pkl", "wb") as f:
            pickle.dump((self.weights, self.biases), f)
        print("Saved network parameters to 'network_params.pkl'.")

    def load_parameters(self):
        """Load weights and biases from a file."""
        with open("..\\cache\\network_params.pkl", "rb") as f:
            self.weights, self.biases = pickle.load(f)
        print("Loaded network parameters from 'network_params.pkl'.")


    