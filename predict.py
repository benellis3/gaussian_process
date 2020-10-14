import logging
from functools import partial

from scipy.optimize import minimize

from gp import GaussianProcess
from kernels import periodic, rbf
from load_data import READ_TIME, TIDE_HEIGHT, process_data
from plot import plot

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def make_gp(tide_height_to_predict, tide_height_data):
    gp = GaussianProcess(
        lambda x, y, params: params["stretch"]
        * (partial(periodic, params=params)(x, y) + partial(rbf, params=params)(x, y)),
    )
    return gp


def fit_gp(gp, params, noise, to_predict, data, x, y):

    gp.sigma = noise
    gp.set_kernel_params(params)
    gp.fit_and_predict(to_predict, data, x, y)
    LOG.info(f"Log Marginal Likelihood: {gp.log_marginal_likelihood}")
    return -gp.log_marginal_likelihood


def fit(x, gp, fit_fun):
    params = {
        "periodic.length": x[0],
        "periodic.p": x[1],
        "rbf.length": x[2],
        "stretch": x[4],
    }
    sigma = x[3]
    return fit_fun(gp, params, sigma)


def main():
    tide_height_data, tide_height_to_predict, true_tide_height = process_data()
    gp = make_gp(tide_height_to_predict, tide_height_data)
    fit_gp_function = partial(
        fit_gp,
        to_predict=tide_height_to_predict.reset_index(),
        data=tide_height_data.reset_index(),
        x=READ_TIME,
        y=TIDE_HEIGHT,
    )

    optimize_fun = partial(fit, gp=gp, fit_fun=fit_gp_function)
    # optimize_fun([0.80965198, 0.34435784, 0.07544029, 0.03377254])
    bounds = [(0.001, 1), (0.1, 1), (0.001, 1), (0.15, 1), (0.01, 1)]
    guess = [0.5, 0.75, 0.05, 0.1, 1]
    optim_result = minimize(optimize_fun, guess, method="L-BFGS-B", bounds=bounds)
    if not optim_result.success:
        LOG.error(f"Optimization Failed: {optim_result.message}")
        return
    # fit the function with the best params again
    LOG.info(f"Final Optimisation Parameters: {optim_result.x}")
    optimize_fun(optim_result.x)
    plot(gp, tide_height_data, true_tide_height, TIDE_HEIGHT)


if __name__ == "__main__":
    main()
