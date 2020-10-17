""" Some useful plotting functions for creating
the final plots
"""
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc("font", family="serif")
plt.rc("xtick", labelsize="x-small")
plt.rc("ytick", labelsize="x-small")
plt.rcParams["figure.figsize"] = (12, 8)


def _join_predictions(data, predictions):
    return pd.concat([data, predictions]).sort_index()


class GPPlot:
    def __init__(
        self,
        data,
        true_data,
        mean,
        var,
        predictions,
        column,
        join=True,
    ):
        self.data = data
        self.true_data = true_data
        self.mean = mean
        self.var = var
        self.predictions = predictions
        self.column = column
        self.join = join

    def init_plot(self):
        self.fig, self.ax = plt.subplots()

    def set_gp(self, mean, var, predictions):
        self.mean = mean
        self.var = var
        self.predictions = predictions

    def plot_predictions(self, num_draws=1):
        for prediction in self.predictions:

            preds = (
                _join_predictions(self.data, prediction) if self.join else prediction
            )

            self.ax.plot(preds.index, preds[self.column], label="Function Draw")

    def plot_uncertainties(self, multiple):
        bars_low = (
            pd.concat([self.data, self.mean - multiple * self.var]).sort_index()
            if self.join
            else self.mean - multiple * self.var
        )

        bars_high = (
            pd.concat([self.data, self.mean + multiple * self.var]).sort_index()
            if self.join
            else self.mean + multiple * self.var
        )

        self.ax.fill_between(
            bars_low.index,
            bars_low[self.column],
            bars_high[self.column],
            label=f"{multiple} $\sigma$",
            alpha=0.5,
        )

    def plot_scatter(self):
        self.ax.scatter(
            self.true_data.index,
            self.true_data[self.column],
            label="True Normalised Tide Height",
            marker="+",
        )
        # plot the observed sensor data
        self.ax.scatter(
            self.data.index,
            self.data[self.column],
            label="Observed Normalised Tide Height",
            marker="+",
        )

    def plot(self, num_draws=1, plot_scatter=True):
        """
        Plot the gp, sensor data, true data and a function draw.
        """
        self.plot_uncertainties(2)
        self.plot_uncertainties(1)

        if plot_scatter:
            self.plot_scatter()

        self.plot_predictions(num_draws=num_draws)
        plt.xlabel("Time")
        plt.ylabel("Normalised Tide Height")
        self.ax.legend(loc="upper left")

    def savefig(self, fig_name="plot"):
        plt.savefig(f"figures/{fig_name}.pdf")


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
