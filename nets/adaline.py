import numpy as np
import pandas as pd
import utils


def train(training_patterns, targets, weights, learning_rate):
    epoch_count = 1

    # =======
    # Step 1 - While stopping condition is false
    # =======
    stop_condition = False
    while not stop_condition:
        print('EPOCH: ', epoch_count)
        epoch_count += 1

        initial_weights = np.copy(weights)
        largest_weight_change = 0

        # =======
        # Step 2 - For each training pair s:t
        # =======
        for i, training_pattern in enumerate(training_patterns):
            print("\tTRAINING PATTERN: ", training_pattern)

            # =======
            # Step 3 - Set activations of input units:
            # =======
            input_units = training_pattern
            print("\t\tINPUT: ", input_units)

            # =======
            # Step 4 - Compute response of output unit
            # =======
            output = utils.compute_net_input_to_output(weights, input_units)
            print("\t\tNET: ", output)

            # =======
            # Step 5 - Update weights and bias
            # =======
            current_target = targets[i]
            print("\t\tTARGET: ", current_target)

            print("\t\tCURRENT WEIGHTS: ", weights)

            # if output != current_target:
            weights = update_weights(weights, learning_rate, current_target, output, training_pattern)
            largest_weight_change = max(get_largest_weight_change_input_pattern(initial_weights, weights), largest_weight_change)

            print("\t\tNEW WEIGHTS: ", weights)
            print("\t\tLARGEST WEIGHT CHANGE:", largest_weight_change)

        # =======
        # Step 6 - Test stop condition
        # =======
        if largest_weight_change < tolerance:
            stop_condition = True


def update_weights(weights, learning_rate, current_target, output, training_pair):
    return weights + learning_rate * (current_target - output) * training_pair


def get_largest_weight_change_input_pattern(initial_weights, weights):
    difference_weights = np.absolute(initial_weights - weights)
    return np.max(difference_weights)


def classify(weights):
    outputs = np.array()
    for i, training_pair in enumerate(batch):
        output = utils.compute_net_input_to_output(weights, training_pair)
        print("\t\tOUTPUT: ", output)
        value = apply_activation_function(output)
        outputs = np.append(outputs, value)
    return outputs

def apply_activation_function(output):
    theta = 0
    if output >= theta:
        return 1
    else:
        return -1

# ======
# Execution
# ======

dataframe = pd.read_csv("resources/and.csv")
utils.insert_bias_dataframe(dataframe)

# Set training patterns
training_patterns = dataframe.iloc[:, :3].to_numpy()

# Set targets
targets = dataframe.iloc[:, 3:].to_numpy()

# Initializa weights
weights = np.array([0, 0, 0])

# Set learning rate (por que?)
learning_rate = 0.1

# Set tolerance (por que?)
tolerance = 0.0555555556

train(training_patterns, targets, weights, learning_rate)