import numpy as np
import pandas as pd
import utils

# =======
# Step 0
# =======

# Initializa weights
weights = np.array([0, 0])

# Initialize bias
bias = 0

# Set learning rate
learning_rate = 1


def train(batch, target, weights, bias, learning_rate):
    epoch_count = 1

    # =======
    # Step 1 - While stopping condition is false
    # =======
    stop_condition = False
    while not stop_condition:
        print('EPOCH: ', epoch_count)
        epoch_count += 1

        initial_bias = bias
        initial_weights = np.copy(weights)

        # =======
        # Step 2 - For each training pair s:t
        # =======
        for i, training_pair in enumerate(batch):
            print("\tTRAINING PATTERN: ", training_pair)

            # =======
            # Step 3 - Set activations of input units:
            # =======
            input_units = training_pair
            print("\t\tINPUT: ", input_units)

            # =======
            # Step 4 - Compute response of output unit
            # =======
            input = utils.compute_net_input(bias, weights, input_units)
            print("\t\tNET: ", input)

            output = apply_activation_function(input)
            print("\t\tOUT: ", output)

            # =======
            # Step 5 - Update weights and bias
            # =======
            current_target = target[i]
            print("\t\tTARGET: ", current_target)

            if output != current_target:
                weights = update_weights(weights, learning_rate, current_target, training_pair)
                bias = update_bias(bias, learning_rate, current_target)

            print("\t\tWEIGHTS: ", weights)
            print("\t\tBIAS: ", bias)

        # =======
        # Step 6 - Test stop condition
        # =======
        if np.array_equal(initial_weights, weights) and initial_bias == bias:
            stop_condition = True


def apply_activation_function(input):
    theta = 0.2
    if input > theta:
        return 1
    elif input >= -theta:
        return 0
    else:
        return -1


def update_weights(weights, learning_rate, current_target, training_pair):
    return weights + learning_rate * current_target * training_pair


def update_bias(bias, learning_rate, current_target):
    return bias + learning_rate * current_target


# ======
# Execution
# ======

dataframe = pd.read_csv("resources/and.csv")

batch = dataframe.iloc[:, :2].to_numpy()
target = dataframe.iloc[:, 2:].to_numpy()

train(batch, target, weights, bias, learning_rate)