import matplotlib.pyplot as plt
import numpy as np

# SOURCE: https://numpy.org/doc/stable/user/absolute_beginners.html
# SOURCE: https://matplotlib.org/stable/tutorials/introductory/usage.html
# SOURCE: https://matplotlib.org/stable/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py

np.random.seed(1)  # seed the random number generator.

# Data: a = array ranging 0 to 50, b = 50 random ints, 0-50, d = 50 random samples of the norm dist
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)  # Data: b = pseudo-random linear scaling of a
data['d'] = np.abs(data['d']) * 100  # Data: d = transformation of the norm dist data

# Produces a figure (the entire image) and a axes (the plotting space)
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# Draws the data on the axes and labels
ax.scatter('a', 'b', c='c', s='d', data=data)
ax.set_xlabel('entry a')
ax.set_ylabel('entry b')
# Displays
fig.show()
