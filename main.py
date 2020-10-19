import logging
from argparse import ArgumentParser

import gif
import matplotlib.pyplot as plt

from load_data import TIDE_HEIGHT, chunk, process_data
from plot import GPPlot, plot_scatter
from predict import sequential_predictions, train

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def plot_scatter_command(args):
    data, to_predict, true_data, tide_height_nans = process_data(normalise_data=False)
    fig, ax = plt.subplots()
    plot_scatter(
        ax,
        data,
        true_data,
        TIDE_HEIGHT,
        savefig=args.save_figures,
        fig_name=args.fig_name,
    )


def estimate_noise_command(args):
    data, _, true_data, tide_height_nans = process_data(normalise_data=False)
    true_data = true_data.loc[~tide_height_nans.values]
    diff = true_data - data
    LOG.info(f"Estimate of the noise: {diff.std()}")


def train_command(args):
    data, to_predict, true_data, tide_height_nans = process_data(normalise_data=True)
    predictions, mean, var, _ = train(
        to_predict,
        data,
    )
    # filter the true tide_height to only be
    # at the non_nan points
    true_data_filtered = true_data.loc[tide_height_nans.values]
    plot = GPPlot(
        data,
        true_data_filtered,
        mean,
        var,
        [predictions],
        TIDE_HEIGHT,
    )
    plot.init_plot()
    plot.plot()
    if args.save_figures:
        plot.savefig(args.fig_name)


def sequential_prediction_command(args):
    data, to_predict, true_data, tide_height_nans = process_data(normalise_data=True)
    max_time = data.index.max()
    means = []
    variances = []
    predictions = []
    data_chunks = []
    for data_chunk in chunk(data):
        prediction, mean, var = sequential_predictions(data_chunk, max_time=max_time)
        means.append(mean)
        variances.append(var)
        predictions.append(prediction)
        data_chunks.append(data_chunk)

    @gif.frame
    def animate(i):
        plot = GPPlot(
            data_chunks[i],
            true_data,
            means[i],
            variances[i],
            [predictions[i]],
            TIDE_HEIGHT,
            join=False,
        )
        plot.init_plot()
        plot.plot()

    frames = [animate(i) for i in range(len(means))]

    gif.save(frames, f"{args.fig_name}.gif", duration=60, unit="s", between="startend")


def main(args):
    parser = ArgumentParser(
        description="Plot a scatter plot of data or fit a GP and plot the result"
    )
    parser.add_argument(
        "--save-figures", action="store_true", help="Save the generated figures as pdf"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Do not show any of the generated plots"
    )
    parser.add_argument("--fig-name", default="plot", help="A name for the plot")
    subparsers = parser.add_subparsers()
    parser_scatter = subparsers.add_parser(
        "plot_scatter", help="Generates a scatter plot of the data"
    )
    parser_scatter.set_defaults(func=plot_scatter_command)

    parser_train = subparsers.add_parser(
        "train", help="Trains a GP and optimises its hyperparameters on all the data"
    )
    parser_train.set_defaults(func=train_command)

    parser_seq = subparsers.add_parser(
        "sequential_prediction",
        help="Generates predictions for a GP on a sequential basis",
    )
    parser_seq.set_defaults(func=sequential_prediction_command)

    parser_noise = subparsers.add_parser(
        "estimate_noise", help="Estimate the noise in the data"
    )
    parser_noise.set_defaults(func=estimate_noise_command)

    parsed_args = parser.parse_args(args)
    parsed_args.func(parsed_args)
    if not parsed_args.quiet:
        plt.show()


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
