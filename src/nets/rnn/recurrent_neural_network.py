import numpy as np

from src.nets.activationfunction.binary_sigmoid import BinarySigmoid
from src.nets.utils import utils
#from src.nets.exemplo.plotting import plot_error as pt

class RecurrentNeuralNetwork:

    def __init__(self, neurons_per_layer, learning_rate=0.1, error_tolerance=0.1, weights_matrix_list=None,
                 grammar=None, stop_condition_element=False):
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.error_tolerance = error_tolerance
        self.weights_matrix_list = weights_matrix_list if weights_matrix_list else self.create_weights_matrix_list()
        self.activation_function = BinarySigmoid()
        self.epoch_count = 0
        self.layers_input = list()
        self.layers_output = list()
        self.dict_grammar = grammar
        self.stop_condition_element = stop_condition_element
        self.mean_squared_error_list = list()

    def create_weights_matrix_list(self):
        weights_matrix_list = list()
        last_amount_neurons = self.neurons_per_layer[0] + self.neurons_per_layer[1]
        for i, amount_neurons in enumerate(self.neurons_per_layer):
            if i != 0:
                weights_matrix_list.append(np.zeros((amount_neurons, last_amount_neurons + 1)))
                last_amount_neurons = amount_neurons
        return weights_matrix_list

    def train(self, string_list):
        self.epoch_count = 0
        for string in string_list:
            self.epoch_count += 1

            symbol_id = 0
            symbol_list = string
            symbol_encoded_list = list(map(self.get_encoder, symbol_list))

            context_layer = [0.5] * self.neurons_per_layer[1]
            stop_condition = False
            total_squared_error_string = list()
            while not stop_condition:
                current_symbol_encoded = symbol_encoded_list[symbol_id]
                symbol_id += 1
                target_symbol = symbol_list[symbol_id]
                target_symbol_encoded = symbol_encoded_list[symbol_id]

                bias_input_layer = [1]
                input_units = np.concatenate((bias_input_layer, current_symbol_encoded, context_layer), axis=0)
                output = self.feedforward(input_units)

                weight_correction = self.backpropagation(output, target_symbol_encoded)
                self.update_weights(weight_correction)

                total_squared_error_symbol = self.calculate_total_squared_error(output, target_symbol_encoded)
                #print('Total Squared Error Symbol: ', total_squared_error_symbol)
                total_squared_error_string.append(total_squared_error_symbol)

                if self.stop_condition_element == target_symbol:
                    stop_condition = True
                else:
                    context_layer = np.copy(self.get_output_last_hidden_layer())

            self.mean_squared_error_list.append(np.mean(total_squared_error_string))

    def feedforward(self, input_units):
        self.layers_input.append(utils.remove_bias_layer(input_units))
        self.layers_output.append(utils.remove_bias_layer(input_units))

        for i, weights in enumerate(self.weights_matrix_list):
            layer_input = self.compute_layer_input(weights, input_units)
            self.layers_input.append(layer_input)

            layer_output = self.activation_function.apply_activation_function(np.copy(layer_input))
            self.layers_output.append(layer_output)

            input_units = utils.insert_bias_inputs(layer_output)

        return layer_output

    def get_output_last_hidden_layer(self):
        return self.layers_output[len(self.layers_output) - 2]

    def backpropagation(self, output, current_target):
        weight_corrections = list()

        # 1 - Calcular variacao dos pesos entre output_layer e hidden_layer
        error_information_weight_adjustment = (current_target - output) * self.activation_function.apply_derivate(self.layers_input[2])

        error_information_weight_adjustment_transpose = np.transpose([error_information_weight_adjustment])
        weight_correction = self.learning_rate * error_information_weight_adjustment_transpose * utils.insert_bias_inputs(self.layers_output[1])

        weight_corrections.insert(0, weight_correction)

        # 2 - Calcular variacao pesos entre input_layer e hidden_layer
        error_correction_weight_adjustment_input = error_information_weight_adjustment_transpose * utils.get_weights_layer_without_bias(self.weights_matrix_list[1])[0]
        error_information_weight_adjustment = error_correction_weight_adjustment_input * self.activation_function.apply_derivate(self.layers_input[1])
        error_information_weight_adjustment = np.sum(error_information_weight_adjustment, axis=0)

        weight_variation_per_error_correction = list()
        for error_correction in error_information_weight_adjustment:
            weight_variation_per_error_correction.append(self.learning_rate * error_correction * utils.insert_bias_inputs(self.layers_output[0]))

        weight_corrections.insert(0, np.array(weight_variation_per_error_correction))

        return weight_corrections

    def compute_layer_input(self, weights, input_units):
        return np.dot(weights, input_units)

    def update_weights(self, weight_correction):
        for i, weights in enumerate(self.weights_matrix_list):
            self.weights_matrix_list[i] = weights + weight_correction[i]

    def calculate_total_squared_error(self, output, target):
        return np.sum((output - target) ** 2)

    def get_letter(self, vector):
        return [x for x in self.dict_grammar if self.dict_grammar[x] == vector]

    def get_encoder(self, letter):
        return self.dict_grammar.get(letter)

    #def plot(self):
    #    pt.plot_error(self.mean_squared_error_list, "Test")
