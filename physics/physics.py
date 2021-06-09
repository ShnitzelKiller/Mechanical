import igl
from meshplot import plot
import numpy as np

V = np.array([
    [0., 0, 0],
    [1, 0, 0],
    [1, 1, 1],
    [2, 1, 0]
])

F = np.array([
    [0, 1, 2],
    [1, 3, 2]
])

plot(V, F)