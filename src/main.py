import json
import os

from src.nets.rnn.recurrent_neural_network import RecurrentNeuralNetwork

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
#
# dataframe = pd.read_csv("resources/xor_bipolar_input_bipolar_target.csv")
# utils.insert_bias_dataframe(dataframe)
#
# # Set training patterns
# training_patterns = dataframe.iloc[:, :3].to_numpy()
#
# # Set targets
# targets = dataframe.iloc[:, 3:].to_numpy().flatten()
#
# learning_rate = 0.2
#
# error_tolerance = 0.05
#
# weights_input_hidden = np.array([
#     [-0.3378, 0.1970, 0.3099],
#     [0.2771, 0.3191, 0.1904],
#     [0.2859, -0.1448, -0.0347],
#     [-0.3329, 0.3594, -0.4861]
# ])
# weights_hidden_output = np.array([[-0.1401, 0.4919, -0.2913, -0.3979, 0.3581]])
# weights_list = [weights_input_hidden, weights_hidden_output]
#
# mlp = MultilayerPerceptron(training_patterns, targets, [2, 4, 1], learning_rate, error_tolerance, None)
# mlp.train()

###########
# Recurrent Neural Network
###########

learning_rate = 0.0125

error_tolerance = 0.05

grammar = {
    'B': [1, 0, 0, 0, 0, 0],
    'E': [1, 0, 0, 0, 0, 0],
    'S': [0, 1, 0, 0, 0, 0],
    'P': [0, 0, 1, 0, 0, 0],
    'T': [0, 0, 0, 1, 0, 0],
    'V': [0, 0, 0, 0, 1, 0],
    'X': [0, 0, 0, 0, 0, 1]
}

stop_condition = 'E'

rnn = RecurrentNeuralNetwork([6, 3, 6], learning_rate=learning_rate, error_tolerance=error_tolerance, grammar=grammar,
                             stop_condition_element=stop_condition)

# string_amount = 6000
# string_list = [gs.generate_string()[2].__add__(['E']) for i in range(string_amount)]

folder_name = os.path.dirname(os.path.abspath(__file__))
filename = os.path.join(folder_name, 'nets/exemplo/grammar.json')
with open(filename) as json_file:
    string_list = json.load(json_file)

# Treinando os mesmos pesos
rnn.train(string_list)

# Usando os mesmos pesos
# weights_z_v = [
#     [-0.70241423, 2.90520929, -1.46287391, -0.39782865, -0.65885591, -2.14346943, -0.39650197, -0.11675449,
#      1.05002266, 1.2385648, ],
#     [-0.77968576, -3.23834314, 5.74947794, -2.84805611, 0.51617198, -2.21781518, 2.1550897, -2.90109297,
#      -0.47355646, 6.69954713],
#     [-0.9549995, 1.95759764, 1.05795331, -2.41193727, -2.80322437, 3.61191549, -2.70926095, 2.34288057,
#      -1.12546089, -0.88138996]
# ]
#
# weights_y_w = [
#     [-5.74661671, -3.8605457, 4.89642468, 4.00437399], [-3.36472465, -0.69471095, 4.07332745, -3.83618402],
#     [-1.85311539, 0.80541688, -6.1195417, 2.0184946], [-0.0192946, 3.12163234, -4.75455344, -3.38343167],
#     [2.79194852, -3.32606222, -4.18600693, -2.57267547], [-3.46751078, -0.89217647, 4.72746952, -3.48326156]
# ]
#
# weights = [weights_z_v, weights_y_w]
# rnn.weights_matrix_list = weights

rnn.test([['B', 'T', 'X', 'S', 'E'], ['B', 'P', 'T', 'V', 'V', 'E']])
