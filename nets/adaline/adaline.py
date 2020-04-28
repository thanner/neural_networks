import numpy as np

from nets.utils import utils
from nets.utils.others.neural_network_graph import NeuralNetworkGraph


class Adaline:

    def __init__(self, training_patterns, targets, weights, theta=0, learning_rate=0.1, tolerance=0.1):
        self.training_patterns = training_patterns
        self.targets = targets
        self.weights = weights
        self.theta = theta
        self.learning_rate = learning_rate
        self.tolerance = tolerance
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

            initial_weights = np.copy(self.weights)
            largest_weight_change = 0

            # =======
            # Step 2 - For each training pair s:t
            # =======
            for i, training_pattern in enumerate(self.training_patterns):
                print("\tTRAINING PATTERN: ", training_pattern)

                # =======
                # Step 3 - Set activations of input units:
                # =======
                input_units = training_pattern

                # =======
                # Step 4 - Compute response of output unit
                # =======
                output = utils.compute_net_input(self.weights, input_units)
                print("\t\tNET: ", output)

                # =======
                # Step 5 - Update weights and bias
                # =======
                current_target = self.targets[i]
                print("\t\tTARGET: ", current_target)

                print("\t\tCURRENT WEIGHTS: ", self.weights)

                self.weights = self.update_weights(current_target, output, training_pattern)
                largest_weight_change = max(self.get_largest_weight_change_input_pattern(initial_weights),
                                            largest_weight_change)

                print("\t\tNEW WEIGHTS: ", self.weights)
                print("\t\tLARGEST WEIGHT CHANGE:", largest_weight_change)

            # =======
            # Step 6 - Test stop condition
            # =======
            if largest_weight_change < self.tolerance:
                stop_condition = True

    def update_weights(self, current_target, output, training_pair):
        return self.weights + self.learning_rate * (current_target - output) * training_pair

    def get_largest_weight_change_input_pattern(self, initial_weights):
        difference_weights = np.absolute(initial_weights - self.weights)
        return np.max(difference_weights)

    def print_graph(self):
        training_patterns_without_bias = self.training_patterns[:, 1:]
        line_values = utils.calculate_line_values(self.weights[0], self.weights[1], self.weights[2],
                                                  training_patterns_without_bias)
        neural_network_graph = NeuralNetworkGraph(self.epoch_count, training_patterns_without_bias,
                                                  NeuralNetworkGraph.target_representation(self.targets), line_values)
        neural_network_graph.plot_graph()
