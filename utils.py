import numpy as np


def compute_net_input(bias, weights, input_units):
    return bias + np.sum(weights * input_units)


def compute_net_input_to_output(weights, input_units):
    return np.sum(weights * input_units)


def insert_bias_dataframe(dataframe):
    dataframe.insert(0, "Bias", 1)
