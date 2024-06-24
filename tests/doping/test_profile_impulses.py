import numpy as np
import pytest

from gdevsim.doping.profile_impulses import (
    constant_impulse,
    gaussian_impulse,
    skewed_gaussian_impulse,
)


@pytest.mark.parametrize("ymax, ymax_offset, y_depth, expected_results", [
    (5, -1, 3, [1, 1, 1, 0, 0, 0]),  # Basic case
    (5, 0, 5, [0, 0, 0, 0, 0, 0]),   # Entire range is doped
    (3, -2, 1, [0, 0, 1, 0, 0, 0]),  # Narrow doping window
    (10, -5, 10, [1, 1, 1, 1, 1, 1]) # ymax out of range of y values
])
def test_constant_impulse_varied(data_regression, ymax, ymax_offset, y_depth, expected_results):
    # Define test inputs
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 1, 2, 3, 4, 5])

    # Call the function under test
    result = constant_impulse(x, y, ymax, ymax_offset, y_depth)

    # Use data_regression fixture to check the result
    data_regression.check({
        "x": x.tolist(),
        "y": y.tolist(),
        "result": result.tolist(),
        "expected": expected_results
    })


@pytest.mark.parametrize("x, y, ymax, range, vertical_straggle, lateral_straggle, expected_results", [
    (np.linspace(-1, 1, 5), np.linspace(3, 5, 5), 5, 1, 1, 1, [0.8824969, 0.93941306, 1, 0.93941306, 0.8824969]),  # Fine grid around center
    (np.array([0]), np.array([6]), 5, 1, 1, 1, [0]),  # Above ymax
    (np.linspace(-2, 2, 5), np.array([4]*5), 5, 1, 1, 1, [0.04393693, 0.32465247, 0.60653066, 0.32465247, 0.04393693]),  # Lateral spread
    (np.array([0]), np.linspace(2, 4, 3), 5, 1, 1, 1, [0.60653066, 0.8824969, 0.93941306])  # Vertical spread
])
def test_gaussian_impulse(data_regression, x, y, ymax, range, vertical_straggle, lateral_straggle, expected_results):
    result = gaussian_impulse(x, y, ymax, range, vertical_straggle, lateral_straggle)
    data_regression.check({
        "x": x.tolist(),
        "y": y.tolist(),
        "result": result.tolist(),
        "expected": expected_results
    })


@pytest.mark.parametrize("x, y, ymax, range, vertical_straggle, vertical_skew, lateral_straggle, expected_results", [
    (np.linspace(-1, 1, 5), np.linspace(3, 5, 5), 5, 1, 1, 0.1, 1, [0.8824969, 0.93941306, 1, 0.93941306, 0.8824969]),  # Fine grid around center
    (np.array([0]), np.array([6]), 5, 1, 1, 0.1, 1, [0]),  # Above ymax
    (np.linspace(-2, 2, 5), np.array([4]*5), 5, 1, 1, 0.1, 1, [0.04393693, 0.32465247, 0.60653066, 0.32465247, 0.04393693]),  # Lateral spread
    (np.array([0]), np.linspace(2, 4, 3), 5, 1, 1, 0.1, 1, [0.60653066, 0.8824969, 0.93941306])  # Vertical spread
])
def test_skewed_gaussian_impulse(data_regression, x, y, ymax, range, vertical_straggle, vertical_skew, lateral_straggle, expected_results):
    result = skewed_gaussian_impulse(x, y, ymax, range, vertical_straggle, vertical_skew, lateral_straggle)
    data_regression.check({
        "x": x.tolist(),
        "y": y.tolist(),
        "result": result.tolist(),
        "expected": expected_results
    })
