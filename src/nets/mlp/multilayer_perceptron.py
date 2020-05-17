import numpy as np

from src.nets.activationfunction.bipolar_sigmoid import BipolarSigmoid
from src.nets.utils import utils


class MultilayerPerceptron:

    def __init__(self, training_patterns, targets, neurons_per_layer, learning_rate=0.1, error_tolerance=0.1,
                 weights_matrix_list=None):
        self.training_patterns = training_patterns
        self.targets = targets
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.error_tolerance = error_tolerance
        self.weights_matrix_list = weights_matrix_list if weights_matrix_list else self.create_weights_matrix_list()
        self.activation_function = BipolarSigmoid()
        self.epoch_count = 0
        self.input_net = list()
        self.output_net = list()

    def create_weights_matrix_list(self):
        weights_matrix_list = list()
        last_amount_neurons = self.neurons_per_layer[0]
        for i, amount_neurons in enumerate(self.neurons_per_layer):
            if i != 0:
                weights_matrix_list.append(np.random.rand(amount_neurons, last_amount_neurons + 1))
            last_amount_neurons = amount_neurons
        return weights_matrix_list

    def train(self):
        self.epoch_count = 0
        stop_condition = False
        while not stop_condition:
            self.epoch_count += 1
            training_pair_errors = list()
            for i, training_pair in enumerate(self.training_patterns):
                self.input_net = list()
                self.output_net = list()
                output = self.feedforward(training_pair)
                current_target = self.targets[i]
                weight_correction = self.backpropagation(output, current_target)
                if output != current_target:
                    self.update_weights(weight_correction)
                training_pair_errors.append((output - current_target) ** 2)
            total_squared_error = self.calculate_total_squared_error(training_pair_errors)
            print('epoch:', self.epoch_count, '| total_squared_error', total_squared_error)
            if total_squared_error < self.error_tolerance or self.epoch_count > 1000:
                stop_condition = True

    def feedforward(self, training_pair):
        input_units = training_pair
        self.input_net.append(utils.remove_bias_layer(input_units))
        self.output_net.append(utils.remove_bias_layer(input_units))
        for i, weights in enumerate(self.weights_matrix_list):
            layer_input = self.compute_layer_input(weights, input_units)
            self.input_net.append(layer_input)
            layer_output = self.activation_function.apply_activation_function(np.copy(layer_input))
            self.output_net.append(layer_output)
            input_units = utils.insert_bias_inputs(layer_output)
        return layer_output

    def backpropagation(self, output, current_target):
        weight_correction = list()

        # 1 - Calcular variacao dos pesos entre output_layer e hidden_layer
        error_correction_weight_adjustment = (current_target - output) * self.activation_function.apply_derivate(
            self.input_net[2])
        weight_correction.insert(0, self.learning_rate * error_correction_weight_adjustment * utils.insert_bias_inputs(
            self.output_net[1]))

        # 2 - Calcular variacao pesos entre input_layer e hidden_layer
        error_correction_weight_adjustment_input = error_correction_weight_adjustment * \
                                                   utils.get_weight_layer_without_bias(self.weights_matrix_list, 1)[0]
        error_correction_weight_adjustment = error_correction_weight_adjustment_input * self.activation_function.apply_derivate(
            self.input_net[1])
        weight_variation_per_error_correction = list()
        for error_correction in error_correction_weight_adjustment:
            weight_variation_per_error_correction.append(
                self.learning_rate * error_correction * utils.insert_bias_inputs(self.output_net[0]))
        weight_correction.insert(0, np.array(weight_variation_per_error_correction))

        return weight_correction

    def compute_layer_input(self, weights, input_units):
        return np.dot(weights, input_units)

    def update_weights(self, weight_correction):
        for i, weights in enumerate(self.weights_matrix_list):
            self.weights_matrix_list[i] = weights + weight_correction[i]

    def calculate_total_squared_error(self, training_pair_errors):
        return np.sum(training_pair_errors)
