import copy
import math

import numpy as np

import src.nets.exemplo.generate_string as gs

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
learning_rate_alpha = 0.2
momentum_mi = 0.999
tetha = 0
error_tollerance = 0.05


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


def applying_MLP_BP(training_input_vector_s, target_output_vector_y_t, training_input_vector_s_letters,
                    target_output_vector_y_t_letters, input_layer_x, hidden_layer_z, z_in, y_in,
                    error_information_term_delta_y, weight_correction_term_Delta_Z,
                    previous_weight_correction_term_Delta_Z, weight_correction_term_Delta_Y,
                    previous_weight_correction_term_Delta_Y, weights_z_v, weights_y_w, learning_rate_alpha, n, p, m,
                    TSE, MSE, momentum_mi):
    for s in range(30):
        (inputStringD, outputStringD, inputStringL, outputStringL) = gs.generate_string()
        training_input_vector_s.append(copy.deepcopy(inputStringD))
        target_output_vector_y_t.append(copy.deepcopy(outputStringD))
        training_input_vector_s_letters.append(copy.deepcopy(inputStringL))
        target_output_vector_y_t_letters.append(copy.deepcopy(outputStringL))
        squared_error_layer_y.append(copy.deepcopy(inputStringD))
        context_layer_c = [0.5, 0.5, 0.5]
        print('training_input_vector_s', training_input_vector_s)
        print('training_input_vector_s_letters', training_input_vector_s_letters[s])
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
            print('output_layer_y', round(output_layer_y[1], 1), round(output_layer_y[2], 1),
                  round(output_layer_y[3], 1), round(output_layer_y[4], 1), round(output_layer_y[5], 1),
                  round(output_layer_y[6], 1))
            # Update context units
            for j in range(1, p + 1):
                context_layer_c[j - 1] = hidden_layer_z[j]


weights_z_v = np.random.uniform(-0.5, 0.5, (4, 10))
weights_z_v[0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
weights_z_v[1] = [-0.70241423, 2.90520929, -1.46287391, -0.39782865, -0.65885591, -2.14346943, -0.39650197, -0.11675449,
                  1.05002266, 1.2385648, ]
weights_z_v[2] = [-0.77968576, -3.23834314, 5.74947794, -2.84805611, 0.51617198, -2.21781518, 2.1550897, -2.90109297,
                  -0.47355646, 6.69954713]
weights_z_v[3] = [-0.9549995, 1.95759764, 1.05795331, -2.41193727, -2.80322437, 3.61191549, -2.70926095, 2.34288057,
                  -1.12546089, -0.88138996]
weights_y_w = np.random.uniform(-0.5, 0.5, (7, 4))
weights_y_w[0] = [0, 0, 0, 0]
weights_y_w[1] = [-5.74661671, -3.8605457, 4.89642468, 4.00437399]
weights_y_w[2] = [-3.36472465, -0.69471095, 4.07332745, -3.83618402]
weights_y_w[3] = [-1.85311539, 0.80541688, -6.1195417, 2.0184946]
weights_y_w[4] = [-0.0192946, 3.12163234, -4.75455344, -3.38343167]
weights_y_w[5] = [2.79194852, -3.32606222, -4.18600693, -2.57267547]
weights_y_w[6] = [-3.46751078, -0.89217647, 4.72746952, -3.48326156]
p = len(hidden_layer_z) - 1
n = len(input_layer_x) - 1 - p
m = len(output_layer_y) - 1
TSE = []
MSE = []
print('-------testing string---------')
applying_MLP_BP(training_input_vector_s, target_output_vector_y_t, training_input_vector_s_letters,
                target_output_vector_y_t_letters, input_layer_x, hidden_layer_z, z_in, y_in,
                error_information_term_delta_y, weight_correction_term_Delta_Z, previous_weight_correction_term_Delta_Z,
                weight_correction_term_Delta_Y, previous_weight_correction_term_Delta_Y, weights_z_v, weights_y_w,
                learning_rate_alpha, n, p, m, TSE, MSE, momentum_mi)
