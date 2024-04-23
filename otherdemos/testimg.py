import matplotlib.pyplot as plt
import numpy as np

# Create a figure and a set of subplots
fig, ax = plt.subplots(figsize=(6, 6))

# Set the limits of the x and y axes
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)

# Set the grid
ax.grid(True, which='both', linestyle='-', linewidth=1)

# Set major ticks for both axes
ax.set_xticks(np.arange(0, 17, 1))
ax.set_yticks(np.arange(0, 10, 1))

# Add labels to the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')

# Set the aspect of the plot to be equal, so the grid will be square
ax.set_aspect('equal', adjustable='box')

plt.title('Grid with Ruler')
plt.show()
