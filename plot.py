"""
Some useful plotting functions for creating
the final plots
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rcParams["figure.figsize"] = (12, 8)


def _join_predictions(data, predictions):
    return pd.concat([data, predictions]).sort_index()


def plot_predictions(ax, predictions, data, column, num_draws=2):
    for prediction in predictions:
        joined_preds = _join_predictions(data, prediction)
        ax.plot(joined_preds.index, joined_preds[column], label="Function Draw")


def plot_uncertainties(ax, mu, sigma, multiple, data, column):
    bars_low = pd.concat([data, mu - multiple * sigma]).sort_index()
    bars_high = pd.concat([data, mu + multiple * sigma]).sort_index()
    ax.fill_between(
        bars_low.index,
        bars_low[column],
        bars_high[column],
        label=f"{multiple} $\sigma$",
        alpha=0.5,
    )


def plot_scatter(ax, data, true_data, column, savefig=False, fig_name="scatter"):
    """
    Plot the scatter plots of the data
    """
    ax.scatter(true_data.index, true_data[column], label=f"True {column}", marker="+")
    ax.scatter(data.index, data[column], label=f"Sensor {column}", marker="+")
    plt.xlabel("Time")
    plt.ylabel(f"{column}")
    ax.legend(loc="best")
    if savefig:
        plt.savefig(f"figures/{fig_name}.pdf")


def plot_lines(ax, splits):
    for i, split in enumerate(splits):
        # This stops the legend filling up with multiple entries for these lines and is
        # super ugly -- much better way is to use a legend artist but life is too short
        if i != 0:
            ax.axvline(split, color="purple", linestyle="--")
        else:
            ax.axvline(split, color="purple", linestyle="--", label="Data Splits")


def plot(
    ax,
    data,
    true_data,
    mean,
    var,
    predictions,
    column,
    num_draws=1,
    savefig=False,
    fig_name="plot",
):
    """
    Plot the gp, sensor data, true data and a function draw.
    """
    plot_uncertainties(ax, mean, var, 2, data, column)
    plot_uncertainties(ax, mean, var, 1, data, column)
    # plot the true data
    ax.scatter(
        true_data.index,
        true_data[column],
        label="True Normalised Tide Height",
        marker="+",
    )
    # plot the observed sensor data
    ax.scatter(
        data.index, data[column], label="Observed Normalised Tide Height", marker="+"
    )
    plot_predictions(ax, predictions, data, column, num_draws=num_draws)
    plt.xlabel("Time")
    plt.ylabel("Normalised Tide Height")
    ax.legend(loc="best")
    if savefig:
        plt.savefig(f"figures/{fig_name}.pdf")
