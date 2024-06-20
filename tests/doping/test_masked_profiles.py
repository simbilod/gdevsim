import numpy as np
import pytest
from gdevsim.doping.masked_profiles import masked_implant_profile_uz

from gdevsim.doping.profile_impulses import gaussian_impulse

@pytest.mark.parametrize("ymax, peak_concentration, x_bounds, range, vertical_straggle, lateral_straggle", [
    (5, 2.0, [(-3, -1), (1, 3)], 1.0, 0.1, 0.1),  # Standard case
    (0, 2.0, [(-3, -1), (1, 3)], 1.0, 0.1, 0.1),  # Zero ymax
    (10, 2.0, [(-3, -1), (1, 3)], 1.0, 0.1, 0.1),  # High ymax
    (5, 0, [(-3, -1), (1, 3)], 1.0, 0.1, 0.1),  # Zero peak concentration
    (5, -1, [(-3, -1), (1, 3)], 1.0, 0.1, 0.1),  # Negative peak concentration
    (5, 2.0, [], 1.0, 0.1, 0.1),  # No bounds
    (5, 2.0, [(0, 2)], 1.0, 0.1, 0.1),  # Single bound
    (5, 2.0, [(-10, -8), (8, 10)], 1.0, 0.1, 0.1),  # Out of normal range bounds
    (5, 2.0, [(-3, -1), (1, 3)], 0.5, 0.05, 0.05),  # Smaller range and straggles
    (5, 2.0, [(-3, -1), (1, 3)], 2.0, 0.2, 0.2),  # Larger range and straggles
])
def test_masked_implant_profile_uz(data_regression, ymax, peak_concentration, x_bounds, range, vertical_straggle, lateral_straggle):
    # Define test inputs
    x = np.linspace(-5, 5, 100)
    y = np.linspace(0, 10, 100)

    # Run the function
    result = masked_implant_profile_uz(
        x=x,
        y=y,
        ymax=ymax,
        impulse_function=lambda x, y, ymax: gaussian_impulse(x, y, ymax, range=range, vertical_straggle=vertical_straggle, lateral_straggle=lateral_straggle),
        peak_concentration=peak_concentration,
        x_bounds=x_bounds
    )

    # Check the results
    data_regression.check({
        "x": x.tolist(),
        "y": y.tolist(),
        "result": result.tolist()
    }, basename=f"test_masked_implant_profile_uz_gaussian_{ymax}_{peak_concentration}_{x_bounds}_{range}_{vertical_straggle}_{lateral_straggle}")
