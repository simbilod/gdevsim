import numpy as np


def identity(data):
    # Does nothing
    return data

def thresholded_log_difference(field, x0, x1, threshold=1E9):
    """Function that returns the absolute log difference between two quantities, if both quantities are above a given value."""
    array = np.abs(np.log10(np.abs(field[x0])) - np.sign(field[x0])*np.sign(field[x1])*np.log10(np.abs(field[x1])))
    array[np.abs(field[x0]) < threshold] = 0
    array[np.abs(field[x1]) < threshold] = 0
    return array

def signed_log(data):
    # Apply logarithm to the absolute values of the data and preserve the sign
    data_with_nan = np.where(data == 0, np.nan, data)
    transformed_data = np.sign(data_with_nan) * np.log10(np.abs(data_with_nan))
    return transformed_data
