import logging
from functools import partial

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from gp import GaussianProcess
from kernels import periodic, rbf
from load_data import READ_TIME, TIDE_HEIGHT

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
SLICES = 10

BOUNDS = [(0.01, None), (0.01, None), (4, None), (0.01, None), (8, None), (0.001, None)]


def make_gp():
    gp = GaussianProcess(
        lambda x, y, params: partial(periodic, params=params)(x, y)
        + params["ratio"] * partial(rbf, params=params)(x, y),
        sigma=0.03,
    )
    return gp


def fit_gp(gp, params, to_predict, data, x, y):
    gp.set_kernel_params(params)
    gp.fit_and_predict(to_predict, data, x, y)
    LOG.info(f"Log Marginal Likelihood: {gp.log_marginal_likelihood}")
    return -gp.log_marginal_likelihood


def fit(x, gp, fit_fun):
    params = {
        "periodic.length": x[0],
        "periodic.p": x[1],
        "periodic.variance": x[2],
        "rbf.length": x[3],
        "rbf.variance": x[4],
        "ratio": x[5],
    }
    return fit_fun(gp, params)


def sequential_predictions(tide_height_data, max_time=2.5):
    bounds = [
        (0.2, 1.0),
        (0.6, 1.0),
        (1, 10),
        (0.01, 0.1),
        (1, 10),
        (0.001, 0.01),
    ]
    guess = [
        np.random.uniform(bound[0], bound[1] if bound[1] else bound[0] * 10)
        for bound in bounds
    ]
    seq_to_predict = np.linspace(tide_height_data.index.min(), max_time, 500)
    seq_to_predict = pd.DataFrame(seq_to_predict, columns=[READ_TIME])
    seq_predictions, seq_mean, seq_var, _ = train(
        seq_to_predict, tide_height_data, initial_guess=guess
    )
    seq_predictions = seq_predictions.set_index(seq_to_predict[READ_TIME].values)
    seq_mean = seq_mean.set_index(seq_to_predict[READ_TIME].values)
    seq_var = seq_var.set_index(seq_to_predict[READ_TIME].values)
    return (seq_predictions, seq_mean, seq_var)


def train(tide_height_to_predict, tide_height_data, initial_guess=None, bounds=None):
    gp = make_gp()
    fit_gp_function = partial(
        fit_gp,
        to_predict=tide_height_to_predict.reset_index(),
        data=tide_height_data.reset_index(),
        x=READ_TIME,
        y=TIDE_HEIGHT,
    )
    bounds = bounds if bounds is not None else BOUNDS
    optimize_fun = partial(fit, gp=gp, fit_fun=fit_gp_function)
    guess = [0.35, 0.69, 4, 0.04, 16, 0.01] if initial_guess is None else initial_guess
    optim_result = minimize(
        optimize_fun,
        guess,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-3, "gtol": 1e-2},
    )
    if not optim_result.success:
        import pdb

        pdb.set_trace()
        raise Exception(f"Optimization Failed: {optim_result.message}")

    # fit the function with the best params again
    LOG.info(f"Final Optimisation Parameters: {optim_result.x}")
    optimize_fun(optim_result.x)
    predictions = gp.predictions
    mean = gp.mean
    var = np.sqrt(np.diag(gp.cov))
    predictions = pd.DataFrame(
        predictions, index=tide_height_to_predict.index, columns=[TIDE_HEIGHT]
    )
    mean = pd.DataFrame(mean, index=tide_height_to_predict.index, columns=[TIDE_HEIGHT])
    var = pd.DataFrame(var, index=tide_height_to_predict.index, columns=[TIDE_HEIGHT])
    return predictions, mean, var, optim_result.x
