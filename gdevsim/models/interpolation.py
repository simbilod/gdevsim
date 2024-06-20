from collections.abc import Callable

import devsim as ds
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
from shapely import Polygon, Point


def add_structured_data_to_mesh(
    device: str,
    region: str,
    name: str,
    x_array: np.array,
    y_array: np.array,
    z_array: np.array,
    val_array: np.array,
) -> None:
    """
    Interpolate data stored on a regular grid onto a DEVSIM model.

    Arguments:
        device: name of the devsim device
        region: name of the devsim region
        name: associated with the model of the data being added
        x_array: [Nx:1] array of x-coordinates of the structured grid
        y_array: [Ny:1] array of y-coordinates of the structured grid
        z_array: [Nz:1] array of z-coordinates of the structured grid
        val_array: [Nx:Ny:Nz] array of model values on the structured model
        dimension_scaling: in case the units of x_array are different than DEVSIM
    """
    if z_array is not None:
        raise NotImplementedError("Interpolation of z-data is not implemented yet!")

    interp = RegularGridInterpolator(
        (x_array, y_array), val_array.T, bounds_error=False, fill_value=0
    )

    if name not in ds.get_node_model_list(device=device, region=region):
        ds.node_solution(device=device, region=region, name=name)

    xpos = np.array(ds.get_node_model_values(device=device, region=region, name="x"))
    ypos = np.array(ds.get_node_model_values(device=device, region=region, name="y"))

    data_on_mesh = interp(np.column_stack((xpos, ypos)))
    ds.set_node_values(device=device, region=region, name=name, values=data_on_mesh)


def add_unstructured_data_to_mesh(
    device: str,
    region: str,
    name: str,
    x_array: np.array,
    y_array: np.array,
    z_array: np.array,
    val_array: np.array,
    val_transformation: Callable,
) -> None:
    """
    Interpolate data stored on a regular grid onto a DEVSIM model.

    Arguments:
        device: name of the devsim device
        region: name of the devsim region
        name: associated with the model of the data being added
        x_array: [N:1] array of x-coordinates of the structured grid
        y_array: [N:1] array of y-coordinates of the structured grid
        z_array: [N:1] array of z-coordinates of the structured grid
        val_array: [N:1] array of model values on the structured model
        dimension_scaling: in case the units of x_array are different than DEVSIM
    """
    if z_array is not None:
        raise NotImplementedError("Interpolation of z-data is not implemented yet!")

    xpos = np.array(ds.get_node_model_values(device=device, region=region, name="x"))
    ypos = np.array(ds.get_node_model_values(device=device, region=region, name="y"))

    data_on_mesh = val_transformation(griddata(
        (x_array, y_array), val_array, xi=np.column_stack((xpos, ypos)),fill_value=0, method="linear"
    ))

    if name not in ds.get_node_model_list(device=device, region=region):
        ds.node_solution(device=device, region=region, name=name)

    ds.set_node_values(device=device, region=region, name=name, values=data_on_mesh)



def add_xy_binary_data_to_mesh(
    device: str,
    region: str,
    name: str,
    polygon: Polygon,
    value: float = 0.0,
) -> None:
    """
    Add binary data (either value or 0 within the xy polygon) to a DEVSIM model.

    Arguments:
        device: name of the devsim device
        region: name of the devsim region
        name: associated with the model of the data being added
        polygon: Polygon,
        value: float = 0.0,
    """

    xpos = np.array(ds.get_node_model_values(device=device, region=region, name="x"))
    ypos = np.array(ds.get_node_model_values(device=device, region=region, name="y"))

    if name not in ds.get_node_model_list(device=device, region=region):
        ds.node_solution(device=device, region=region, name=name)

    data_on_mesh = np.zeros_like(xpos)
    data_on_mesh = np.where(polygon.contains(Point(xpos, ypos)), value, data_on_mesh)

    ds.set_node_values(device=device, region=region, name=name, values=data_on_mesh)