import numpy as np

from src.nets.activationfunction.binary_sigmoid import BinarySigmoid
from src.nets.utils import utils


class RecurrentNeuralNetwork:

    def __init__(self, neurons_per_layer, learning_rate=0.1, error_tolerance=0.1, weights_matrix_list=None,
                 grammar=None, stop_condition_element=False):
        self.neurons_per_layer = neurons_per_layer
        self.learning_rate = learning_rate
        self.error_tolerance = error_tolerance
        self.weights_matrix_list = weights_matrix_list if weights_matrix_list else self.create_weights_matrix_list()
        self.activation_function = BinarySigmoid()
        self.epoch_count = 0
        self.input_layer = list()
        self.hidden_layer = list()
        self.output_layer = list()
        self.dict_grammar = grammar
        self.stop_condition_element = stop_condition_element

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

            # =======
            # Step 1 - Set activations of context units to 0.5
            # =======
            context_layer = [0.5] * self.neurons_per_layer[1]

            # =======
            # Step 2 - Until the end of string
            # =======
            stop_condition = False
            while not stop_condition:
                current_symbol = symbol_list[symbol_id]
                current_symbol_encoded = symbol_encoded_list[symbol_id]
                print("\tInput Symbol:", current_symbol, " - ", current_symbol_encoded)

                symbol_id += 1

                # =======
                # Step 4 - Present successor to output units as target response
                # =======

                target_symbol = symbol_list[symbol_id]
                target_symbol_encoded = symbol_encoded_list[symbol_id]
                print("\tTarget symbol:", target_symbol, " - ", target_symbol_encoded)

                # =======
                # Step 5 - Calculate predicted successor
                # =======

                # initial_weights = np.copy(self.weights_matrix_list)
                # training_pair_errors = list()
                # self.input_layer = list()
                # self.output_layer = list()

                bias_input_layer = [1]
                input_units = np.concatenate((bias_input_layer, current_symbol_encoded, context_layer), axis=0)
                print("\t\tInput units: ", input_units)

                output = self.feedforward(input_units)
                print("\tOutput symbol: ", output)

                # TODO: Parei aqui
                # =======
                # Step 6 - Determine error, backpropagate, update weights
                # =======

                weight_correction = self.backpropagation(output, target_symbol_encoded)

                print("\t\tWEIGHT CORRECTION", weight_correction)

                # TODO: Levar em consideração context_layer_c[j - 1] = hidden_layer_z[j]
                self.update_weights(weight_correction)

                print("\t\tNEW WEIGHTS: ", self.weights_matrix_list)

                training_pair_errors.append((output - current_target) ** 2)

                # =======
                # Step 7 - Test for stop condition
                # =======
                if self.stop_condition_element or self.epoch_count > 1000:
                    stop_condition = True
                else:
                    # TODO: Saída dos hidden layer e nao entrada
                    self.context_layer = self.hidden_layer

    def feedforward(self, input_units):
        # =======
        # Step 3 - Set activations of input units
        # =======

        self.input_layer.append(utils.remove_bias_layer(input_units))
        self.output_layer.append(utils.remove_bias_layer(input_units))

        for i, weights in enumerate(self.weights_matrix_list):
            # =======
            # Step 4 - Compute response of hidden and output units
            # =======

            layer_input = self.compute_layer_input(weights, input_units)
            print(f"\t\tLAYER {i} INPUT: ", layer_input)
            self.input_layer.append(layer_input)

            layer_output = self.activation_function.apply_activation_function(np.copy(layer_input))
            print(f"\t\tLAYER {i} OUTPUT: ", layer_output)
            self.output_layer.append(layer_output)

            input_units = utils.insert_bias_inputs(layer_output)

        return layer_output

    def backpropagation(self, output, current_target):
        weight_correction = list()

        # print("current_target", current_target)
        # print("output", output)

        # 1 - Calcular variacao dos pesos entre output_layer e hidden_layer
        error_correction_weight_adjustment = (current_target - output) * self.activation_function.apply_derivate(
            self.input_layer[2])
        print("error_layer", current_target - output)
        # print("y_in", self.input_layer[2])
        # print("error_information_term_delta_y", error_correction_weight_adjustment)
        # print('---------------')
        # print("hidden_layer_z", utils.insert_bias_inputs(self.output_layer[1]))

        weight_correction.insert(0, self.learning_rate * error_correction_weight_adjustment * utils.insert_bias_inputs(
            self.output_layer[1]))

        # 2 - Calcular variacao pesos entre input_layer e hidden_layer
        error_correction_weight_adjustment_input = error_correction_weight_adjustment * \
                                                   utils.get_weight_layer_without_bias(self.weights_matrix_list, 1)[0]
        error_correction_weight_adjustment = error_correction_weight_adjustment_input * self.activation_function.apply_derivate(
            self.input_layer[1])
        weight_variation_per_error_correction = list()
        for error_correction in error_correction_weight_adjustment:
            weight_variation_per_error_correction.append(
                self.learning_rate * error_correction * utils.insert_bias_inputs(self.output_layer[0]))
        weight_correction.insert(0, np.array(weight_variation_per_error_correction))

        return weight_correction

    def compute_layer_input(self, weights, input_units):
        return np.dot(weights, input_units)

    def update_weights(self, weight_correction):
        for i, weights in enumerate(self.weights_matrix_list):
            self.weights_matrix_list[i] = weights + weight_correction[i]

    def calculate_total_squared_error(self, training_pair_errors):
        return np.sum(training_pair_errors)

    def get_letter(self, vector):
        return [x for x in self.dict_grammar if self.dict_grammar[x] == vector]

    def get_encoder(self, letter):
        return self.dict_grammar.get(letter)
