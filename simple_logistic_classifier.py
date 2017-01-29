

import matplotlib.pyplot as plt
import numpy as np

scores = [1.0, 2.0, 3.0]

def softmax(x):
	return np.exp(x) / np.sum(np.exp(x), axis=0)

print(softmax(scores))

scores = np.array([[1, 2, 3, 6],
                   [2, 4, 5, 6],
                   [3, 8, 7, 6]])

print(softmax(scores))

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()




