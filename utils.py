import numpy as np
import pandas as pd


def compute_net_input(bias, weights, input_units):
    return bias + np.sum(weights * input_units)


def compute_net_input_to_output(weights, input_units):
    return np.sum(weights * input_units)


def insert_bias_dataframe(dataframe):
    dataframe.insert(0, "Bias", 1)


def calculate_line_values(bias, weight_1, weight_2, training_patterns_without_bias):
    x_coordinate = training_patterns_without_bias[:, :1]
    x_higher = np.max(x_coordinate)
    x_lower = np.min(x_coordinate)

    x1 = np.array(range(x_lower - 1, x_higher + 2))
    x2 = - (weight_1 / weight_2) * x1 - (bias / weight_2)
    return pd.DataFrame({'x': x1, 'y': x2})
