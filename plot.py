"""
Some useful plotting functions for creating
the final plots
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()


def _join_predictions(data, predictions):
    return pd.concat([data, predictions]).sort_index()


def plot_predictions(ax, gp, data, column, num_draws=2):
    predictions = [
        pd.DataFrame(gp.draw(), index=gp.to_predict_x, columns=[column])
        for _ in range(num_draws)
    ]
    for i, prediction in enumerate(predictions):
        joined_preds = _join_predictions(data, prediction)
        ax.plot(joined_preds.index, joined_preds[column], label=f"Function Draw {i}")


def plot_uncertainties(ax, mu, sigma, multiple, data, to_predict_index, column):
    bars_low = pd.concat(
        [
            data,
            pd.DataFrame(
                index=to_predict_index, data=mu - multiple * sigma, columns=[column]
            ),
        ]
    ).sort_index()
    bars_high = pd.concat(
        [
            data,
            pd.DataFrame(
                index=to_predict_index, data=mu + multiple * sigma, columns=[column]
            ),
        ]
    ).sort_index()
    ax.fill_between(
        bars_low.index,
        bars_low[column],
        bars_high[column],
        label=f"{multiple} $\sigma$",
        alpha=0.5,
    )


def plot(gp, data, true_data, column, num_draws=1):
    fig, ax = plt.subplots()
    mu = gp.mean
    sigma = np.sqrt(np.diag(gp.cov))

    plot_uncertainties(ax, mu, sigma, 2, data, gp.to_predict_x, column)
    plot_uncertainties(ax, mu, sigma, 1, data, gp.to_predict_x, column)
    # plot the true data
    ax.scatter(
        true_data.loc[gp.to_predict_x].index,
        true_data.loc[gp.to_predict_x, column],
        label=f"True {column}",
        marker="+",
    )
    # plot the observed sensor data
    ax.scatter(data.index, data[column], label=f"Observed {column}", marker="+")
    plot_predictions(ax, gp, data, column, num_draws=num_draws)
    ax.legend(loc="best")
    plt.show()
