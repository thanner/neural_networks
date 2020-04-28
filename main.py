import numpy as np
import pandas as pd

from nets.mlp.multilayer_perceptron import MultilayerPerceptron
from nets.utils import utils

# #############
# # Perceptron With Bias - Ver 67, 68
# #############
#
# dataframe = pd.read_csv("resources/and_binary_input_bipolar_target.csv")
#
# # Set training patterns
# training_patterns = dataframe.iloc[:, :2].to_numpy()
#
# # Set targets
# targets = dataframe.iloc[:, 2:].to_numpy().flatten()
#
# # Initialize bias
# bias = 0
#
# # Initializa weights
# weights = np.array([0, 0])
#
# # Set theta
# theta = 0.2
#
# # Set learning rate
# learning_rate = 1
#
# perceptron = PerceptronWithBias(training_patterns, targets, bias, weights, theta, learning_rate)
# perceptron.train()
# perceptron.print_graph()
#
# ###########
# # Multilayer Perceptron
# ###########
#
# dataframe = pd.read_csv("resources/and_binary_input_bipolar_target.csv")
# utils.insert_bias_dataframe(dataframe)
#
# # Set training patterns
# training_patterns = dataframe.iloc[:, :3].to_numpy()
#
# # Set targets
# targets = dataframe.iloc[:, 3:].to_numpy().flatten()
#
# # Initialize bias
# bias = 0
#
# # Initializa weights
# weights = np.array([0, 0, 0])
#
# # Set theta
# theta = 0.2
#
# # Set learning rate
# learning_rate = 1
#
# perceptron = Perceptron(training_patterns, targets, weights, theta, learning_rate)
# perceptron.train()
# perceptron.print_graph()

# ###########
# # Adaline - 83
# ###########
#
# dataframe = pd.read_csv("resources/and_binary_input_bipolar_target.csv")
# utils.insert_bias_dataframe(dataframe)
#
# # Set training patterns
# training_patterns = dataframe.iloc[:, :3].to_numpy()
#
# # Set targets
# targets = dataframe.iloc[:, 3:].to_numpy().flatten()
#
# # Initializa weights
# weights = np.array([0, 0, 0])
#
# # Set theta
# theta = 0
#
# # Set learning rate (por que?)
# learning_rate = 0.1
#
# # Set tolerance (por que?)
# tolerance = 0.0555555556
#
# adaline = Adaline(training_patterns, targets, weights, theta, learning_rate, tolerance)
# adaline.train()
# adaline.print_graph()

###########
# Multilayer Perceptron
###########

dataframe = pd.read_csv("resources/xor_bipolar_input_bipolar_target.csv")
utils.insert_bias_dataframe(dataframe)

# Set training patterns
training_patterns = dataframe.iloc[:, :3].to_numpy()

# Set targets
targets = dataframe.iloc[:, 3:].to_numpy().flatten()

# Set learning rate
learning_rate = 0.02

mlp = MultilayerPerceptron(training_patterns, targets, [2, 4, 1], learning_rate)
mlp.train()