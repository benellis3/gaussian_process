"""
An implementation of gaussian processes.
Author: Ben Ellis
"""
from functools import partial
from typing import Callable, Dict

import numpy as np
import pandas as pd


class GaussianProcess:
    def __init__(
        self,
        kernel: Callable[[np.array, np.array, Dict[str, float]], np.array],
        sigma: float = 0.001,
    ):
        self.kernel = kernel
        self.sigma = sigma

    def draw(self):
        """
        Given the fitted mean and covariance, draw function values at the fit
        points.
        """
        return np.random.default_rng().multivariate_normal(self.mean, self.cov)

    def set_kernel_params(self, params: Dict[str, float]):
        self.kernel = partial(self.kernel, params=params)

    def fit_and_predict(
        self, to_predict: pd.DataFrame, data: pd.DataFrame, x: str, y: str
    ):
        """
        fits a gaussian process to the data, and produces predictions for the
        time values in to_predict.

        Arguments
        ---------
        to_predict: a Dataframe to predict the data for. This should be a
                    numeric type
        data: The observed data we can use.
        x: The X column to use (i.e. the x we want to use to make the
           prediction)
        y: The Y column to use (i.e. the column we wish to predict)

        Returns
        -------
        arr -- A sample of function values at the to_predict points.
        """
        # compute the kernel function on the data
        self.data_x = np.array(data[x])
        self.data_y = np.array(data[y])
        self.to_predict_x = np.array(to_predict[x])
        # add noise to help with the matrix conditioning
        k_dd = self.kernel(self.data_x, self.data_x) + 1e-4 * np.eye(
            self.data_x.shape[0]
        )
        # and on the data and to_predict
        k_pd = self.kernel(self.to_predict_x, self.data_x)
        k_pp = self.kernel(self.to_predict_x, self.to_predict_x)

        data_mat = k_dd + (self.sigma ** 2) * np.eye(k_dd.shape[0])
        cholesky = np.linalg.cholesky(data_mat)
        # using the pseudoinverse is slower, but is better for conditioning
        inv = np.linalg.pinv(cholesky)

        inv = np.dot(inv.T, inv)
        self.mean = np.dot(np.dot(k_pd, inv), self.data_y)
        self.cov = k_pp - np.dot(np.dot(k_pd, inv), k_pd.T)
        self.log_marginal_likelihood = (
            -0.5 * np.dot(np.dot(self.data_y.T, inv), self.data_y)
            - np.sum(np.log(np.diag(cholesky)))
            - (self.data_x.shape[0] / 2) * np.log(2 * np.pi)
        )
        self.predictions = self.draw()
        return self.predictions
