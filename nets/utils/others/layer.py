class Layer:

    def __init__(self, unit_inputs, weights, theta=0):
        self.weights = weights
        self.unit_inputs = unit_inputs
        self.theta = theta
