import numpy as np

class BinarySigmoid:

    def apply_activation_function(self, value):
        return 1.0 / (1 + np.exp(-value))

    def apply_derivate(self, value):
        f1 = self.apply_activation_function(value)
        return f1 * (1 - f1)

