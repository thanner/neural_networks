from nets.utils import utils


class Neuron:

    def __init__(self, unit_inputs, weights, theta=0):
        self.weights = weights
        self.unit_inputs = unit_inputs
        self.theta = theta

    def compute_net_input(self):
        return utils.compute_net_input(self.weights, self.unit_inputs)

    def apply_bipolar_activation_function(self, input):
        if input > self.theta:
            return 1
        elif input >= -self.theta:
            return 0
        else:
            return -1
