import copy
import math

import numpy as np

import src.nets.exemplo.generate_string as gs
import json

training_input_vector_s = []
training_input_vector_s_letters = []
target_output_vector_y_t = []
target_output_vector_y_t_letters = []
input_layer_x = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
weight_correction_term_Delta_Z = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
previous_weight_correction_term_Delta_Z = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
hidden_layer_z = [1, 0, 0, 0]
z_in = [0, 0, 0, 0]
total_delta_inputs_z = [0, 0, 0, 0]
error_information_term_delta_z = [0, 0, 0, 0]
weight_correction_term_Delta_Y = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
previous_weight_correction_term_Delta_Y = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
context_layer_c = [0.5, 0.5, 0.5]
output_layer_y = [1, 0, 0, 0, 0, 0, 0]
y_in = [0, 0, 0, 0, 0, 0, 0]
error_layer_y = [0, 0, 0, 0, 0, 0, 0]
squared_error_layer_y = []
error_information_term_delta_y = [0, 0, 0, 0, 0, 0, 0]
learning_rate_alpha = 0.0125
test = 'test16'


def activation_function(y_z_in):
    return (1 / (1 + math.exp(-y_z_in)))  # binary sigmoid
    # return ((2 / (1 + math.exp(-y_z_in))) - 1) #bipolar sigmoid


def error_information_term_delta(TSE, x):
    derived = activation_function(x) * (1 - activation_function(x))  # binary sigmoid
    # derived = (0.5 * (1 + activationfunction(x)) * (1 - activationfunction(x))) #bipolar sigmoid
    return (TSE * derived)


def total_input_signal(units, weights_w):
    sum_input_signal = 0
    for i in range(len(units)):
        sum_input_signal = sum_input_signal + (units[i] * weights_w[i])
    return sum_input_signal


def sum_delta_inputs(error_information_term_delta, weights_w, j, m):
    sum_input_signal = 0
    for i in range(1, m + 1):
        sum_input_signal = sum_input_signal + (error_information_term_delta[i] * weights_w[i][j])
    return sum_input_signal


def calculate_total_squared_error(squared_error_layer_y, training_input_vector_s):
    total_squared_error = 0
    for s in range(len(training_input_vector_s)):
        for s_t_pair in range(len(training_input_vector_s[s])):
            for se in range(1, len(squared_error_layer_y[s][s_t_pair])):
                total_squared_error = total_squared_error + squared_error_layer_y[s][s_t_pair][se]
    return total_squared_error


def trainning_MLP_BP(training_input_vector_s, target_output_vector_y_t, training_input_vector_s_letters,
                     target_output_vector_y_t_letters, input_layer_x, hidden_layer_z, z_in, y_in,
                     error_information_term_delta_y, weight_correction_term_Delta_Z,
                     previous_weight_correction_term_Delta_Z, weight_correction_term_Delta_Y,
                     previous_weight_correction_term_Delta_Y, weights_z_v, weights_y_w, learning_rate_alpha, n, p, m,
                     TSE, MSE):
    for s in range(6000):
        string_length = 33
        while string_length >= 33:
            (inputStringBoolean, outputStringBoolean, inputStringLetter, outputStringLetter) = gs.generate_string()
            training_input_vector_s.append(copy.deepcopy(inputStringBoolean))
            target_output_vector_y_t.append(copy.deepcopy(outputStringBoolean))
            training_input_vector_s_letters.append(copy.deepcopy(inputStringLetter))
            target_output_vector_y_t_letters.append(copy.deepcopy(outputStringLetter))
            squared_error_layer_y.append(copy.deepcopy(inputStringBoolean))
            string_length = len(inputStringLetter)
        context_layer_c = [0.5, 0.5, 0.5]
        medium_squared_error = 0

        # TODO: TESTE
        training_input_vector_s[s] = [[1, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0]]
        target_output_vector_y_t[s] = [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]]

        for s_t_pair in range(len(training_input_vector_s[s])):
            # feedforward:
            for i in range(1, n + 1):
                input_layer_x[i] = training_input_vector_s[s][s_t_pair][i]
            for c in range(p):
                input_layer_x[i + c + 1] = context_layer_c[c]
            for j in range(1, p + 1):
                z_in[j] = total_input_signal(input_layer_x, weights_z_v[j])
                hidden_layer_z[j] = activation_function(z_in[j])
            for k in range(1, m + 1):
                y_in[k] = total_input_signal(hidden_layer_z, weights_y_w[k])
                output_layer_y[k] = activation_function(y_in[k])

            #print("Output layer", output_layer_y)

            # backpropagation of TSE:
            for k in range(1, m + 1):
                error_layer_y[k] = (target_output_vector_y_t[s][s_t_pair][k - 1] - output_layer_y[k])
                squared_error_layer_y[s][s_t_pair][k] = error_layer_y[k] ** 2
                error_information_term_delta_y[k] = error_information_term_delta(error_layer_y[k], y_in[k])
                for j in range(0, p + 1):
                    weight_correction_term_Delta_Y[k][j] = (
                                learning_rate_alpha * error_information_term_delta_y[k] * hidden_layer_z[j])

            #print("current_target", target_output_vector_y_t[s][s_t_pair])
            #print("output", output_layer_y)

            #print("error_layer", error_layer_y)
            #print("Y_in", y_in)
            #print("------")
            #print("error_information_term_delta_y", error_information_term_delta_y)
            #print("hidden_layer_z", hidden_layer_z)

            for j in range(1, p + 1):
                total_delta_inputs_z[j] = sum_delta_inputs(error_information_term_delta_y, weights_y_w, j, m)
                error_information_term_delta_z[j] = error_information_term_delta(total_delta_inputs_z[j], z_in[j])
                for i in range(0, n + p + 1):
                    weight_correction_term_Delta_Z[j][i] = (
                                learning_rate_alpha * error_information_term_delta_z[j] * input_layer_x[i])

            # Update weights and biases:
            for k in range(1, m + 1):
                for j in range(0, p + 1):
                    weights_y_w[k][j] = weights_y_w[k][j] + weight_correction_term_Delta_Y[k][j]
            for j in range(1, p + 1):
                for i in range(0, n + p + 1):
                    weights_z_v[j][i] = weights_z_v[j][i] + weight_correction_term_Delta_Z[j][i]
            # Update context units
            for j in range(1, p + 1):
                context_layer_c[j - 1] = hidden_layer_z[j]

            total_squared_error = 0
            for tse in range(1, len(squared_error_layer_y[s][s_t_pair])):
                total_squared_error = total_squared_error + squared_error_layer_y[s][s_t_pair][tse]
            # TSE.append(total_squared_error)
            medium_squared_error = medium_squared_error + total_squared_error
            print('TSE', total_squared_error, '| string:', s, training_input_vector_s_letters[s])  # , training_input_vector_s[s]

        medium_squared_error = medium_squared_error / (s_t_pair + 1)
        MSE.append(medium_squared_error)

def generate_string_list(cases):
    return [case.__add__(['E']) for case in cases]

def save_grammar_json(string_list):
    with open('grammar.json', 'w') as filename:
        json.dump(string_list, filename)

weights_z_v = np.random.uniform(-0.5, 0.5, (4, 10))
weights_z_v = np.zeros((4, 10))  # TODO: TESTE
weights_z_v[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

weights_y_w = np.random.uniform(-0.5, 0.5, (7, 4))
weights_y_w = np.zeros((7, 4))  # TODO: TESTE
weights_y_w[0] = [0, 0, 0, 0]

p = len(hidden_layer_z) - 1
n = len(input_layer_x) - 1 - p
m = len(output_layer_y) - 1
TSE = []
MSE = []
trainning_MLP_BP(training_input_vector_s, target_output_vector_y_t, training_input_vector_s_letters,
                 target_output_vector_y_t_letters, input_layer_x, hidden_layer_z, z_in, y_in,
                 error_information_term_delta_y, weight_correction_term_Delta_Z,
                 previous_weight_correction_term_Delta_Z, weight_correction_term_Delta_Y,
                 previous_weight_correction_term_Delta_Y, weights_z_v, weights_y_w, learning_rate_alpha, n, p, m, TSE,
                 MSE)
#pt.plot_error(MSE, test)
#print(test)
#print('\n weights_z_v: \n', weights_z_v)
#print('\n weights_y_w: \n', weights_y_w)

arquivo1 = open('inputD.txt', 'w')
arquivo2 = open('outputD.txt', 'w')
arquivo3 = open('inputL.txt', 'w')
arquivo4 = open('outputL.txt', 'w')
arquivo1.write(str(training_input_vector_s))
arquivo2.write(str(target_output_vector_y_t))
arquivo3.write(str(training_input_vector_s_letters))
arquivo4.write(str(target_output_vector_y_t_letters))
arquivo1.close()
arquivo2.close()
arquivo3.close()
arquivo4.close()

string_list = generate_string_list(training_input_vector_s_letters)
save_grammar_json(string_list)

print(string_list)
print(MSE)