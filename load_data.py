"""
A module to load the sotonmet data and manipulate it into
a suitable format
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

# Constants representing the column names
READ_TIME = "Reading Date and Time (ISO)"
TIDE_HEIGHT = "Tide height (m)"
AIR_TEMP = "Air temperature (C)"
TRUE_TIDE_HEIGHT = "True tide height (m)"


def load_data(path):
    data = pd.read_csv(path)
    # filter to only the relevant columns
    return data[[READ_TIME, TIDE_HEIGHT, TRUE_TIDE_HEIGHT]]


def normalise(data):
    data.loc[:, READ_TIME] = (
        data[READ_TIME] - data[READ_TIME].min()
    ).dt.total_seconds()
    data = pd.DataFrame(
        scale(data.values), columns=[READ_TIME, TIDE_HEIGHT, TRUE_TIDE_HEIGHT]
    )
    return data


def split(data):
    """
    Takes in the data and returns a training, a test set
    and a ground truth set
    """
    # discard the true tide_heights
    tide_heights = data[[READ_TIME, TIDE_HEIGHT]]
    # only predict the NaN values
    tide_height_nans = tide_heights[TIDE_HEIGHT].isna()
    return (
        tide_heights.loc[~tide_height_nans],
        tide_heights.loc[tide_height_nans],
        data[[READ_TIME, TRUE_TIDE_HEIGHT]],
        tide_height_nans,
    )


def process_data(normalise_data=True):
    data = load_data("data/sotonmet.csv")
    # convert column type
    data.loc[:, READ_TIME] = pd.to_datetime(data.loc[:, READ_TIME])
    if normalise_data:
        data = normalise(data)
    (
        tide_height_data,
        tide_height_to_predict,
        true_tide_height,
        tide_height_nans,
    ) = split(data)
    tide_height_to_predict = tide_height_to_predict.set_index(READ_TIME)
    tide_height_data = tide_height_data.set_index(READ_TIME)
    true_tide_height = true_tide_height.set_index(READ_TIME)
    # rename the true tide_height's column
    true_tide_height = true_tide_height.rename(columns={TRUE_TIDE_HEIGHT: TIDE_HEIGHT})
    return tide_height_data, tide_height_to_predict, true_tide_height, tide_height_nans
