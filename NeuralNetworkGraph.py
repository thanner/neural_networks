import matplotlib.pyplot as plt
import numpy as np


class NeuralNetworkGraph:

    def __init__(self, epoch_number, position_points, point_values, line_values):
        self.epoch_number = epoch_number
        self.position_points = position_points
        self.point_values = point_values
        self.line_values = line_values

    def plot_graph(self):
        self.set_title()
        self.set_points()
        self.set_line()
        # Add X and Y Label
        plt.xlabel('x1')
        plt.ylabel('x2')
        # Add a grid
        plt.grid(alpha=.4, linestyle='--')
        # Add a Legend
        plt.legend()
        plt.show()

    def set_title(self):
        plt.title(f"Epoch: {self.epoch_number}")

    def set_line(self):
        x = self.line_values.iloc[:, :1].to_numpy()
        y = self.line_values.iloc[:, 1:].to_numpy()
        plt.plot(x, y, label='Boundary Line')

    def set_points(self):
        for i, self.position_point in enumerate(self.position_points):
            self.set_text(self.position_point[0], self.position_point[1], self.point_values[i])

    def set_text(self, x, y, text):
        text_font = {'color': 'red', 'size': 16}
        plt.text(x, y, text, fontdict=text_font, horizontalalignment='center',
                 verticalalignment='center')

    @staticmethod
    def target_representation(values):
        converter = lambda t: '+' if t > 0 else '-'
        vfunc = np.vectorize(converter)
        return vfunc(values)
