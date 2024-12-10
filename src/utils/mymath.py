import numpy as np

def derivative_central(x, y):
    xp = np.pad(x, 1, mode='edge')
    yp = np.pad(y, 1, mode='edge')
    return (yp[2:] - yp[:-2]) / (xp[2:] - xp[:-2])