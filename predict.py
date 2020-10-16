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


def sequential_predictions(tide_height_to_predict, tide_height_data):
    # split the data into SLICES chunks
    # set the hyperparams to something fairly decent to start with
    min_time = tide_height_data.index.min()
    time_range = tide_height_data.index.max() - min_time
    step_size = time_range / SLICES
    mean = []
    var = []
    predictions = []
    times = []
    guess = None
    for time in np.arange(
        min_time + step_size, tide_height_data.index.max() + step_size, step_size
    ):
        # get the data that is before this time
        seq_data = tide_height_data.loc[min_time:time]
        seq_to_predict = np.linspace(time, time + step_size, 150)
        seq_to_predict = pd.DataFrame(seq_to_predict, columns=[READ_TIME])
        seq_predictions, seq_mean, seq_var, new_guess = train(
            seq_to_predict, seq_data, initial_guess=guess
        )
        guess = new_guess
        seq_predictions = seq_predictions.set_index(seq_to_predict[READ_TIME].values)
        seq_mean = seq_mean.set_index(seq_to_predict[READ_TIME].values)
        seq_var = seq_var.set_index(seq_to_predict[READ_TIME].values)
        mean.append(seq_mean)
        var.append(seq_var)
        predictions.append(seq_predictions)
        times.append(time)
    predictions = pd.concat(predictions).sort_index()
    mean = pd.concat(mean).sort_index()
    var = pd.concat(var).sort_index()
    times.append(times[-1] + step_size)
    return (predictions, mean, var, times)


def train(
    tide_height_to_predict,
    tide_height_data,
    initial_guess=None,
):
    gp = make_gp()
    fit_gp_function = partial(
        fit_gp,
        to_predict=tide_height_to_predict.reset_index(),
        data=tide_height_data.reset_index(),
        x=READ_TIME,
        y=TIDE_HEIGHT,
    )

    optimize_fun = partial(fit, gp=gp, fit_fun=fit_gp_function)
    bounds = [
        (0.01, None),
        (0.01, None),
        (4, None),
        (0.01, None),
        (8, None),
        (0.001, None),
    ]
    guess = [0.35, 0.69, 4, 0.04, 16, 0.01] if initial_guess is None else initial_guess
    optim_result = minimize(optimize_fun, guess, method="L-BFGS-B", bounds=bounds)
    if not optim_result.success:
        LOG.error(f"Optimization Failed: {optim_result.message}")
        return
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
