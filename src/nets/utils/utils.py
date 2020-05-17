import numpy as np
import pandas as pd


def compute_net_input_with_bias(bias, weights, input_units):
    return bias + np.sum(weights * input_units)


def compute_net_input(weights, input_units):
    return np.sum(weights * input_units)


def insert_bias_dataframe(dataframe):
    dataframe.insert(0, "Bias", 1)


def insert_bias_inputs(array):
    return np.insert(array, 0, 1, axis=0)


def remove_bias_layer(array):
    return array[1:]


def calculate_line_values(bias, weight_1, weight_2, training_patterns_without_bias):
    x_coordinate = training_patterns_without_bias[:, :1]

    x_lower = np.min(x_coordinate)
    x_higher = np.max(x_coordinate)

    # Get list of values in coordinate x1 from -1 to +1
    x1 = np.array(range(x_lower - 1, x_higher + 2))

    # Get list of values in coordinate x2 (corresponding to x1) from -1 to +1
    x2 = - (weight_1 / weight_2) * x1 - (bias / weight_2)
    return pd.DataFrame({'x': x1, 'y': x2})


def get_weights_layer_without_bias(weights_layer):
    return weights_layer[:, 1:]


def mean_squared_error(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()
