import numpy as np
import matplotlib.pyplot as plt

# Create the vectors X and Y
x = np.array(range(100))
y = x ** 2

# Create the plot
plt.plot(x, y, label='y = x**2')

# Add a title
plt.title('My first Plot with Python')

# Add X and y Label
plt.xlabel('x axis')
plt.ylabel('y axis')

# Add a grid
plt.grid(alpha=.4, linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()