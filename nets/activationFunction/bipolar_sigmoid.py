import numpy as np

class BipolarSigmoid:

    def apply_activation_function(self, value):
        return (2.0 / (1 + np.exp(-value))) - 1

    def apply_derivate(self, value):
        f2 = self.apply_activation_function(value)
        return (1 / 2) * (1 + f2) * (1 - f2)
