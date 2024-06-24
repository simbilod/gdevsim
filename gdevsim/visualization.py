from pathlib import Path

import matplotlib as mpl
import matplotlib.colors as mcolors
import meshio
import pyvista as pv

from gdevsim.config import PATH


def get_distinguishable_colors(register: bool = False):
    """From https://en.wikipedia.org/wiki/Help:Distinguishable_colors"""
    colors = [
        # "#000000",
        "#FFFF00",
        "#1CE6FF",
        "#FF34FF",
        "#FF4A46",
        "#008941",
        "#006FA6",
        "#A30059",
        "#FFDBE5",
        "#7A4900",
        "#0000A6",
        "#63FFAC",
        "#B79762",
        "#004D43",
        "#8FB0FF",
        "#997D87",
        "#5A0007",
        "#809693",
        "#FEFFE6",
        "#1B4400",
        "#4FC601",
        "#3B5DFF",
        "#4A3B53",
        "#FF2F80",
    ]

    cmap = mcolors.ListedColormap(colors)
    if register:
        try:
            mpl.colormaps.register(name="distinguishable_colors", cmap=cmap)
            return cmap
        except ValueError:
            return cmap


def plot_device(
    field: str = "regions",
    cmap: str = "distinguishable_colors",
    filepath: Path = PATH.temp / "temp_device.dat",
):
    # Need to convert the data from DEVSIM to something meshio can read
    if cmap == "distinguishable_colors":
        get_distinguishable_colors(register=True)
    mesh = meshio.read(str(filepath), file_format="tecplot")
    pv_mesh = pv.wrap(mesh)
    plotter = pv.Plotter(notebook=True)
    plotter = pv.Plotter(window_size=(1200, 1000))
    if field == "regions":
        annotations = {value[0]: key for key, value in mesh.field_data.items()}
        plotter.add_mesh(
            pv_mesh,
            scalars="vtkBlockColors",
            cmap=cmap,
            line_width=1.0,
            lighting=False,
            show_edges=True,
            edge_color="white",
            annotations=annotations,
        )
        plotter.remove_scalar_bar()
        plotter.add_scalar_bar(
            position_x=0.1, position_y=0.30, width=0.8, height=0.1, n_labels=0
        )
        plotter.view_xy()
        (
            (x, y, z),
            (center_x, center_y, center_z),
            (up_x, up_y, up_z),
        ) = plotter.camera_position
        plotter.camera_position = (
            (x, y, z),
            (center_x, center_y, center_z),
            (up_x, up_y, up_z),
        )
        plotter.show()
