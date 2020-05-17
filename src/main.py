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
rnn.train([['B', 'T', 'X', 'S', 'E']])
