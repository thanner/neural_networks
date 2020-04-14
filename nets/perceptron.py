import numpy as np

import utils
from NeuralNetworkGraph import NeuralNetworkGraph


class Perceptron:

    def __init__(self, training_patterns, targets, bias, weights, theta=0, learning_rate=0.1):
        self.training_patterns = training_patterns
        self.targets = targets
        self.bias = bias
        self.weights = weights
        self.theta = theta
        self.learning_rate = learning_rate
        self.epoch_count = 0

    def train(self):
        self.epoch_count = 0

        # =======
        # Step 1 - While stopping condition is false
        # =======
        stop_condition = False
        while not stop_condition:
            self.epoch_count += 1
            print('EPOCH: ', self.epoch_count)

            initial_bias = self.bias
            initial_weights = np.copy(self.weights)

            # =======
            # Step 2 - For each training pair s:t
            # =======
            for i, training_pair in enumerate(self.training_patterns):
                print("\tTRAINING PATTERN: ", training_pair)

                # =======
                # Step 3 - Set activations of input units:
                # =======
                input_units = training_pair
                print("\t\tINPUT: ", input_units)

                # =======
                # Step 4 - Compute response of output unit
                # =======
                input = utils.compute_net_input(self.bias, self.weights, input_units)
                print("\t\tNET: ", input)

                output = self.apply_bipolar_activation_function(input)
                print("\t\tOUT: ", output)

                # =======
                # Step 5 - Update weights and bias
                # =======
                current_target = self.targets[i]
                print("\t\tTARGET: ", current_target)

                if output != current_target:
                    self.update_weights(current_target, training_pair)
                    self.update_bias(current_target)

                print("\t\tWEIGHTS: ", self.weights)
                print("\t\tBIAS: ", self.bias)

            # =======
            # Step 6 - Test stop condition
            # =======
            if np.array_equal(initial_weights, self.weights) and initial_bias == self.bias:
                stop_condition = True

    def apply_bipolar_activation_function(self, input):
        if input > self.theta:
            return 1
        elif input >= -self.theta:
            return 0
        else:
            return -1

    def update_weights(self, current_target, training_pair):
        self.weights = self.weights + self.learning_rate * current_target * training_pair

    def update_bias(self, current_target):
        self.bias = self.bias + self.learning_rate * current_target

    def print_graph(self):
        line_values = utils.calculate_line_values(self.bias, self.weights[0], self.weights[1],
                                                  self.training_patterns)
        neural_network_graph = NeuralNetworkGraph(self.epoch_count, self.training_patterns,
                                                  NeuralNetworkGraph.target_representation(self.targets), line_values)
        neural_network_graph.plot_graph()
