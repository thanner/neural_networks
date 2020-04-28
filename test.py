import nets.utils.utils as utils
import numpy as np

y_true = np.array([2, 3])
y_pred = np.array([8, 4])

print(utils.mean_squared_error(y_true, y_pred))