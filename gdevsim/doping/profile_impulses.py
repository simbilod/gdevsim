"""
Reference: Selberherr, S. (1984). Process Modeling. In: Analysis and Simulation of Semiconductor Devices. Springer, Vienna. https://doi.org/10.1007/978-3-7091-8752-4_3
"""

import numpy as np
import scipy


def constant_impulse(x: np.ndarray, y: np.ndarray, ymax: float, ymax_offset: float, y_depth: float) -> np.ndarray:
    """Constant doping across the window. The material is assumed to start at y=ymax.

    Arguments:
        x: x-coordinates where to evaluate the impulse
        y: y-coordinates where to evaluate the impulse
        ymax: maximum y-coordinate of the profile (target layer ymax is used)
        ymax_offset: for the constant profile, offset of the top of the doping from ymax
        y_depth: the minimum y-coordinate of the profile is ymax - y_depth
    """
    return np.where((y > ymax + ymax_offset) & (y < ymax - y_depth), 0, 1)

def gaussian_impulse(x: np.ndarray, y: np.ndarray, ymax: float, range: float, vertical_straggle: float, lateral_straggle: float) -> np.ndarray:
    """Impulse function representing a gaussian implant concentration resulting from implantation at a single x-location. The material is assumed to start at y=ymax, and so values above are clipped."""
    impulse = np.exp(
        -np.power(x, 2.0) / (2 * np.power(lateral_straggle, 2.0))
    ) * np.exp(
        -np.power(-y - range + ymax, 2.0) / (2 * np.power(vertical_straggle, 2.0))
    )
    return np.where(y > ymax, 0, impulse)


def skewed_gaussian_impulse(
    x: np.ndarray, y: np.ndarray, ymax: float, range: float, vertical_straggle: float, vertical_skew: float, lateral_straggle: float
) -> np.ndarray:
    """
    Returns skewed two half-gaussian implantation profile for dopant in silicon. Valid for |skew| <~ 1. The material is assumed to start at y=ymax, and so values above are clipped.
    """

    def range_eq(Rm, sigma1, sigma2):
        return Rm + np.sqrt(2 / np.pi) * (sigma2 - sigma1)

    def vertical_straggle_eq(Rm, sigma1, sigma2):
        return np.sqrt(
            (sigma1**2 - sigma1 * sigma2 + sigma2**2)
            - 2 / np.pi * (sigma2 - sigma1) ** 2
        )

    def vertical_skew_eq(Rm, sigma1, sigma2):
        return (
            np.sqrt(2 / np.pi)
            * (sigma2 - sigma1)
            * (
                (4 / np.pi - 1) * (sigma1**2 + sigma2**2)
                + (3 - 8 / np.pi) * sigma1 * sigma2
            )
            / vertical_straggle_eq(Rm, sigma1, sigma2) ** 3
        )

    def system(x):
        Rm, sigma1, sigma2 = x
        return [
            range_eq(Rm, sigma1, sigma2) - range,
            vertical_straggle_eq(Rm, sigma1, sigma2) - vertical_straggle,
            vertical_skew_eq(Rm, sigma1, sigma2) - vertical_skew,
        ]

    Rm, sigma1, sigma2 = scipy.optimize.fsolve(
        system, [range, 0.9 * vertical_straggle, 1.1 * vertical_straggle]
    )

    impulse = np.exp(-np.power(x, 2.0) / (2 * np.power(lateral_straggle, 2.0))) * (
        np.reciprocal(np.sqrt(2 * np.pi) * (sigma1 + sigma2) * 1e-4)
        * np.where(
            y < Rm,
            np.exp(-((y - Rm) ** 2) / (2 * sigma1**2)),
            np.exp(-((y - Rm) ** 2) / (2 * sigma2**2)),
        )
    )
    return np.where(y > ymax, 0, impulse)


def pearsonIV_impulse(x, y, ymax, filepath):
    raise NotImplementedError("pearsonIV_impulse not implemented yet, but would be useful to have!")

def sampled_impulse_from_file(x, y, ymax, filepath):
    raise NotImplementedError("Impulse sampled from filepath not implemented yet, but would be useful to have!")



function_mapping = {
    "constant_impulse": constant_impulse,
    "gaussian_impulse": gaussian_impulse,
    "skewed_gaussian_impulse": skewed_gaussian_impulse,
    "pearsonIV_impulse": pearsonIV_impulse,
    "sampled_impulse_from_file": sampled_impulse_from_file
}
