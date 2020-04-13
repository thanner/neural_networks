import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class NeuralNetworkGraph:

    def __init__(self, epoch_number, position_points, point_values, line_values):
        self.plot_title(epoch_number)
        self.plot_points(position_points, point_values)
        self.plot_line(line_values)
        # Add X and Y Label
        plt.xlabel('x1')
        plt.ylabel('x2')
        # Add a grid
        plt.grid(alpha=.4, linestyle='--')
        # Add a Legend
        plt.legend()
        # Show Graph
        plt.show()

    def plot_title(self, epoch_number):
        plt.title(f"Epoch: {epoch_number}")

    def plot_line(self, line_values):
        x = line_values.iloc[:, :1].to_numpy()
        y = line_values.iloc[:, 1:].to_numpy()
        plt.plot(x, y, label='Boundary Line')

    def plot_points(self, position_points, point_values):
        for i, position_point in enumerate(position_points):
            self.plot_text(position_point[0], position_point[1], point_values[i])

    def plot_text(self, x, y, text):
        text_font = {'color': 'red', 'size': 16}
        plt.text(x, y, text, fontdict=text_font, horizontalalignment='center',
                 verticalalignment='center')


def target_binary_representation(values):
    converter = lambda t: '+' if t > 0 else '-'
    vfunc = np.vectorize(converter)
    return vfunc(values)


# Create points
position_points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([0, 0, 0, 1])
targets = target_binary_representation(targets)

# Create the vectors X and Y
x = y = np.array(range(-1, 2))
line_values = pd.DataFrame({'x': x, 'y': y})

graph = NeuralNetworkGraph(1, position_points, targets, line_values)