from collections.abc import Callable

import numpy as np
import scipy
from gdsfactory.typings import LayerStack

from gdevsim.doping.parse_layer_stack_dopings import deserialize_profile


def masked_implant_profile_uz(
    x: np.ndarray,
    y: np.ndarray,
    ymax: float,
    impulse_function: Callable,
    peak_concentration: float = 1.0,
    x_bounds: tuple[tuple[float, float]] = None,
) -> np.ndarray:
    """
    This function generates an implant profile over a domain from an impulse and a mask.

    Parameters:
    x (array-like): x-domain where to define the profile.
    y (array-like): y-domain where to define the profile.
    ymax (float): y-value defining the origin of the profile
    impulse_function (Callable): The impulse function to be used for the implantation profile. Must accept X, Y, and ymax as arguments.
    peak_concentration (float, optional): The peak concentration of the implant. Defaults to 1.0.
    x_bounds (tuple of tuples, optional): List of bounds for the mask openings. Defaults to None (no masking).

    Returns:
        array-like: The calculated implant profile.
    """
    X, Y = np.meshgrid(
        x - np.mean(x), y
    )  # Shift x to be centered at 0 for the convolution to work well

    if x_bounds is None:
        window = np.ones_like(x)
    else:
        window = np.zeros_like(x)
        for x_bound in x_bounds:
            x1, x2 = x_bound
            window += np.heaviside(x - x1, 1) - np.heaviside(x - x2, 1)

    impulse = impulse_function(X, Y, ymax)
    convolved_impulse = scipy.ndimage.convolve1d(impulse, window, axis=1)

    return convolved_impulse / np.max(convolved_impulse) * peak_concentration


def project_profiles_uz(
    layer_stack_geometry: LayerStack,
    doping_data: dict,
    rel_xbuffer: float = 0.2,
    rel_ybuffer: float = 0.2,
    Nx: int = 1000,
    Ny: int = 200,
    doping_fields: tuple = ("acceptor", "donor"),
) -> dict:
    """Project the combined doping profiles onto a structured grid having roughly the coordinates of the target regions.

    TODO:
        * Use a dataclass for doping_data instead of a dict
        * Unit test

    Args:
        doping_data (dict): The doping data to be projected.
        rel_xbuffer (float, optional): how much (relative to x-size of domain) to increase all considered domains in the x-direction for doping structured data
        rel_ybuffer (float, optional): how much (relative to y-size of domain) to increase all considered domains in the y-direction for doping structured data
        Nx (int, optional): Number of points for the x array coordinates
        Ny (int, optional): Number of points for the y array coordinates
    Returns:
        total_profiles: a dictionary where each key is a target region. The value for each key is another dictionary.
        This nested dictionary has keys for each implant type and two special keys: "x_samplings" and "y_samplings".
        The "x_samplings" and "y_samplings" keys correspond to numpy arrays representing the x and y coordinates of the grid points capturing the doping profile.
        The keys corresponding to implant types map to 2D numpy arrays representing the doping profile for that implant type in the target region.
    """

    total_profiles = {}
    for _doping_layer_name, doping_data_dict in doping_data.items():
        # Get all target regions for this layer
        target_regions = doping_data_dict["into"]

        for target_region in target_regions:
            # Make sure it is present
            if target_region in layer_stack_geometry.layers:
                # Always need to reestablish ymax per region
                ymax = (
                    layer_stack_geometry.layers[target_region].zmin
                    + layer_stack_geometry.layers[target_region].thickness
                )

                # Initialize the functions if needed
                if target_region not in total_profiles.keys():
                    total_profiles[target_region] = {}

                    # Figure out how large the x-domain should be given this mask+target
                    bounds = doping_data_dict["domain_bounds"][target_region]
                    if bounds:
                        target_xmin, target_xmax = min(np.array(bounds).flatten()), max(
                            np.array(bounds).flatten()
                        )
                        xbuffer = rel_xbuffer * (target_xmax - target_xmin)
                        target_xmin -= xbuffer
                        target_xmax += xbuffer

                        # Figure out how large y-domain should be given this target
                        target_ymin, target_ymax = sorted(
                            [
                                layer_stack_geometry.layers[target_region].zmin,
                                layer_stack_geometry.layers[target_region].zmin
                                + layer_stack_geometry.layers[target_region].thickness,
                            ]
                        )
                        ybuffer = rel_ybuffer * (target_ymax - target_ymin)
                        target_ymin -= ybuffer
                        target_ymax_domain = target_ymax + ybuffer

                        # Create the sampling arrays
                        sampling_x_array = np.linspace(target_xmin, target_xmax, Nx)
                        total_profiles[target_region]["x_samplings"] = sampling_x_array
                        sampling_y_array = np.linspace(target_ymin, target_ymax_domain, Ny)
                        total_profiles[target_region]["y_samplings"] = sampling_y_array

                # Accumulate the implants
                if doping_data_dict["domain_bounds"][target_region]:
                    for implant_type, implant_profile_str, peak_concentration in zip(
                        doping_data_dict["ion_types"],
                        doping_data_dict["impulse_profiles"],
                        doping_data_dict["peak_concentrations"],
                    ):
                        implant_profile = deserialize_profile(implant_profile_str)
                        # Generate the profile from the domain, mask bounds, and doping impulse
                        if implant_type not in total_profiles[target_region]:
                            total_profiles[target_region][implant_type] = np.zeros([Ny, Nx])
                        total_profiles[target_region][
                            implant_type
                        ] += masked_implant_profile_uz(
                            x=total_profiles[target_region]["x_samplings"],
                            y=total_profiles[target_region]["y_samplings"],
                            ymax=ymax,
                            impulse_function=implant_profile,
                            peak_concentration=peak_concentration,
                            x_bounds=doping_data_dict["mask_bounds"],
                        )

                # handle case where no dopings
                for field in doping_fields:
                    if field not in total_profiles[target_region]:
                        total_profiles[target_region][field] = np.zeros([Ny, Nx])

    return total_profiles
