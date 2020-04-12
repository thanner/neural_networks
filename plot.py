import matplotlib.pyplot as plt
import numpy as np

# Create the vectors X and Y
x = np.array(range(-1, 2))
y = x

# Create the plot
plt.plot(x, y, label='Boundary Line')

# Plot points
text_font = {'color': 'red', 'size': 16}

def plot_text(x, y, text):
    plt.text(x, y, text, fontdict=text_font, horizontalalignment='center',
             verticalalignment='center')

plot_text(0, 0, "-")
plot_text(0, 1, "-")
plot_text(1, 0, "-")
plot_text(1, 1, "+")

# Add a title
plt.title('Epoch')

# Add X and Y Label
plt.xlabel('x1')
plt.ylabel('x2')

# Add a grid
plt.grid(alpha=.4, linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()
