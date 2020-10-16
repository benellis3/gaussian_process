"""
An implementation of some basic kernel functions
"""
import numpy as np
from scipy.spatial.distance import cdist


def rbf(x, x_dash, params):
    """Evaluate the RBF kernel pairwise to construct a covariance matrix"""
    # add a new axis to x and take advantage of broadcasting
    length = params["rbf.length"]
    diff = -cdist(x[:, np.newaxis], x_dash[:, np.newaxis], "sqeuclidean") / length ** 2
    return params["rbf.variance"] * np.exp(diff)


def periodic(x, x_dash, params):
    """Evaluate a periodic kernel function pairwise to construct a covariance matrix"""
    p = params["periodic.p"]
    length = params["periodic.length"]
    diff = np.pi * cdist(x[:, np.newaxis], x_dash[:, np.newaxis], "euclidean") / p
    sin_diff = -(2 / length ** 2) * np.sin(diff) ** 2
    return params["periodic.variance"] * np.exp(sin_diff)
