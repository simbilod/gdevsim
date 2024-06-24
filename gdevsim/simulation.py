import shutil
import sys
import warnings
from pathlib import Path
from typing import Literal

import devsim as ds
import dill
import gdsfactory as gf
import meshio
import numpy as np
import pandas as pd
import pyvista as pv
import yaml
from devsim import set_parameter
from gdsfactory.typings import Callable, Dict, LayerStack, Tuple
from gplugins.common.base_models.component import LayeredComponentBase
from gplugins.common.types import GFComponent
from gplugins.gmsh import get_mesh
from pydantic import NonNegativeFloat, PrivateAttr
from scipy.interpolate import griddata

from gdevsim import ramp
from gdevsim.config import PATH
from gdevsim.doping.masked_profiles import project_profiles_uz
from gdevsim.doping.parse_layer_stack_dopings import (
    parse_layer_stack_doping_uz,
    parse_layer_stack_doping_xy,
)
from gdevsim.logger import Tee
from gdevsim.materials.materials import (
    get_all_materials,
    get_default_physics,
    get_global_parameters,
)
from gdevsim.meshing.get_devsim_from_mesh import devsim_device_from_mesh
from gdevsim.models import models
from gdevsim.models.assign_models import (
    assign_contact_models,
    assign_doping_profiles_uz,
    assign_doping_profiles_xy,
    assign_interface_models,
    assign_region_models,
)
from gdevsim.models.interpolation import (
    add_structured_data_to_mesh,
    add_unstructured_data_to_mesh,
)
from gdevsim.utils.get_component_with_effective_layers import (
    get_component_with_net_layers,
)
from gdevsim.utils.operations import identity
from gdevsim.visualization import get_distinguishable_colors


class DevsimComponent(LayeredComponentBase):
    """
    Represents a component in the DEVSIM simulation environment.

    Relevant parent arguments:
        component: GDSfactory Component to simulate
        layer_stack: description of layers associated with simulation
        pad_xy: how much in-plane space around the component to add for simulation purposes
        pad_z_inner: how much to extend the lower layer of the simulation in the vertical direction for simulation purposes
        pad_z_outer: how much to extend the upper layer of the simulation in the vertical direction for simulation purposes
        wafer_layer: which GDS layer to use to represent WAFER (background) polygons

    DEVSIM-specific arguments:
        mesh_type: type of meshing to perform
            uz: 2D out-of-plane cross-section of the Component between points u1 = (x1, y1) and u2 = (x2, y2)
            xy: 2D in-plane cross-section of the Component at higher z
            3D: full 3D mesh
        uz_mesh_xsection_bounds: if mesh_type == "uz", represents coordinates ((x1, y1),(x2, y2))
        xy_mesh_z: if mesh_type == "xy", represents z-plane
        port_contacts: dict with keys contact names to use when calling simulator, and values the Component port names to consider as terminals
    """
    class Config:
        extra = "allow"

    # Component settings
    component: GFComponent
    layer_stack: LayerStack

    # Mesh settings
    mesh_type: Literal["uz", "xy", "3D"]
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None
    z: float | None = None
    resolutions: Dict = {}
    default_characteristic_length: float = 1.0
    contact_delimiter: str = "@"
    interface_delimiter: str = "___"

    # Terminal renaming
    _port_names_to_contact_names: dict[str, str] | None = PrivateAttr()

    # Fields and physics
    materials_parameters: Dict = get_all_materials()
    global_parameters: Dict = get_global_parameters()
    physics_parameters: Dict = get_default_physics()

    # Extra space around Component
    pad_xy_inner: NonNegativeFloat = 1.0
    pad_xy_outer: NonNegativeFloat = 1.0
    pad_z_inner: float = 0.0
    pad_z_outer: NonNegativeFloat = 0.0
    wafer_layer: tuple[int, int] = (99999, 0)

    # Default filenames
    log_filename: str = "simulation.log"
    error_filename: str = "simulation.err"
    mesh_filename: str = "mesh.msh2"
    device_data_filename: str = "device_data.dat"
    settings_filename: str = "settings.pkl"
    simulation_gds_filename: str = "device.gds"

    def __init__(self, **data):
        super().__init__(**data)
        if "port_names_to_contact_names" not in data:
            self._port_names_to_contact_names = {
                port.name: port.name for port in self.component.get_ports_list()
            }
        else:
            self._port_names_to_contact_names = data["port_names_to_contact_names"]


    @property
    def save_directory(self):
        return PATH.simulation / str(self.__hash__())

    @property
    def device_name(self):
        return str(self.__hash__())

    @property
    def simulation_inputs(self):
        return self.get_component_with_net_layers()

    def contact_interfaces_from_contact(self, contact):
        """Returns all devsim contacts that contain the contact region.

        Arguments:
            contact: given contact name
        """
        contact_region = self.simulation_inputs[2][contact]
        devsim_contacts = list(ds.get_contact_list(device=self.device_name))
        return [contact for contact in devsim_contacts if contact_region in contact]

    def contact_parameters_from_contact(self, contact):
        """Returns all devsim contact parameters that contain the contact region.

        Arguments:
            contact: given contact name
        """
        contact_region = self.simulation_inputs[2][contact]
        contact_parameters = [x for x in ds.get_parameter_list(device=self.device_name) if "bias" in x]
        return [contact for contact in contact_parameters if contact_region in contact]

    def get_component_with_net_layers(self):
        """Computes the temporary Component + LayerStack representing contacts, as well as the mapping between contact_name and DEVSIM contact name."""

        if self._port_names_to_contact_names is None:
            port_names = [
                port.name
                for port in self.component.get_ports_list()
            ]
            self._port_names_to_contact_names = {
                port_name: port_name for port_name in port_names
            }
        else:
            port_names = list(self._port_names_to_contact_names.keys())

        simulation_component, simulation_layerstack, port_map = get_component_with_net_layers(
            component=self.component,
            layer_stack=self.layer_stack,
            port_names=port_names,
            delimiter=self.contact_delimiter,
            new_layers_init=(10010, 0),
            add_to_layerstack=True,
            additional_layers= [self.wafer_layer]
        )

        contact_name_to_simulation_port_map = {}
        for port_name, port_physical in port_map.items():
            contact_name_to_simulation_port_map[self._port_names_to_contact_names[port_name]] = port_physical

        return simulation_component, simulation_layerstack, contact_name_to_simulation_port_map

    def reset(self):
        """Delete all files in the save_directory."""
        for item in self.save_directory.glob("*"):
            if item.is_dir():
                item.rmdir()
            else:
                item.unlink()
        if self.save_directory.exists():
            self.save_directory.rmdir()

    def clean_intermediate_files(self):
        """Deletes all intermediary/temporary files in the save_directory."""
        for file in self.save_directory.glob("*.msh2"):
            file.unlink(missing_ok=True)
        for file in self.save_directory.glob("*.dat"):
            file.unlink(missing_ok=True)
        for file in self.save_directory.glob("*.pos"):
            file.unlink(missing_ok=True)


    def initialize(
        self,
        # Mesh parameters
        threads_available: int = 6,
        resolutions: Dict | None = None,
        remeshing_file: Path | None = None,
        global_scaling: float = 1e-4,  # gdsfactory is in um, semiconductor physics in cm
        default_characteristic_length: float | None = None,
        # Physics settings (for possible override)
        materials_parameters: Dict | None = None,
        global_parameters: Dict | None = None,
        physics_parameters: Dict | None = None,
        # Save settings
        reset_save_directory: bool = False,
        append_to_log: bool = False,
        print_log: bool = True,
        # Below parameters have good defaults, change at your own risk!
        # Solver settings
        device_name: str | None = None,
        device_mesh_name: str | None = None,
        # Save files
        save_directory: Path | str = None,
        log_filename: str | None = None,
        error_filename: str | None = None,
        mesh_filename: str | None = None,
        device_data_filename: str | None = None,
        settings_filename: str | None = None,
        simulation_gds_filename: str | None = None,
    ):
        """Initialize the DevsimComponent simulation."""

        return initialize(
            component=self.simulation_inputs[0],
            layer_stack=self.simulation_inputs[1],
            mesh_type=self.mesh_type,
            xsection_bounds=self.xsection_bounds,
            z=self.z,
            resolutions=resolutions or self.resolutions,
            remeshing_file=remeshing_file,
            global_scaling=global_scaling,
            default_characteristic_length=default_characteristic_length or self.default_characteristic_length,
            wafer_padding=self.pad_xy,
            wafer_layer=self.wafer_layer,
            interface_delimiter=self.interface_delimiter,
            contact_delimiter=self.contact_delimiter,
            threads_available=threads_available,
            materials_parameters=materials_parameters or self.materials_parameters,
            global_parameters=global_parameters or self.global_parameters,
            physics_parameters=physics_parameters or self.physics_parameters,
            device_name=device_name or self.device_name,
            device_mesh_name=device_mesh_name or self.device_name,
            reset_save_directory = reset_save_directory,
            print_log = print_log,
            append_to_log = append_to_log,
            save_directory=save_directory or self.save_directory,
            log_filename=log_filename or self.log_filename,
            error_filename=error_filename or self.error_filename,
            mesh_filename=mesh_filename or self.mesh_filename,
            device_data_filename=device_data_filename or self.device_data_filename,
            settings_filename=settings_filename or self.settings_filename,
            simulation_gds_filename=simulation_gds_filename or self.simulation_gds_filename
        )


    def solve(
        self,
        save_directory: Path | None = None,
        device_data_filename: str | None = None,
        extended_solver: bool = True,
        extended_model: bool = True,
        extended_equation: bool = True,
        threads_available: int = 18,
        threads_task_size=2048,
        absolute_error=1e10,
        relative_error=1e-10,
        maximum_iterations=100,
    ):
        """Initialize the DevsimComponent simulation."""

        save_directory = save_directory or self.save_directory
        device_data_filename = device_data_filename or self.device_data_filename

        return solve(
            save_directory=save_directory,
            device_data_filename=device_data_filename,
            extended_solver=extended_solver,
            extended_model=extended_model,
            extended_equation=extended_equation,
            threads_available=threads_available,
            threads_task_size=threads_task_size,
            absolute_error=absolute_error,
            relative_error=relative_error,
            maximum_iterations=maximum_iterations,
        )

    def load(self, file: str | Path | None = None):
        """Loads simulation state from a file.

        Arguments:
            file: filename. If None, will be save_directory / device_data_filename.
        """
        if file is None:
            file = self.save_directory / self.device_data_filename
        ds.reset_devsim()
        ds.load_devices(file=file)

        # Also load parameters
        parameters_file = file.with_suffix(".yaml")
        if parameters_file.exists():
            with open(parameters_file) as yaml_file:
                parameters_dict = yaml.safe_load(yaml_file)
                for key, params in parameters_dict.items():
                    # Global
                    if key == self.device_name:
                        for param, value in params.items():
                            ds.set_parameter(device=self.device_name, name=param, value=value)
                    # Regional
                    else:
                        for param, value in params.items():
                            ds.set_parameter(device=self.device_name, region=key, name=param, value=value)

        return self


    def write(self,
              file: str | Path | None = None,
              type: str = "devsim",
              include: Tuple[str] = ("*.",),
              exclude: Tuple[str] = ("",),
              ):
        """Writes the current simulation state to a file.

        Arguments:
            file: filename. If None, will be save_directory / device_data_filename.
            type: of data.
            include: list of regex strings determining which fields to save. Default to save all.
            exclude: list of regex strings determining which fields to ignore. Default to not excluding anything.
        """
        if file is None:
            file = self.save_directory / self.device_data_filename
        # ds.write_devices(file=file, device=self.device_name, type=type, include=include, exclude=exclude)
        ds.write_devices(file=file, device=self.device_name, type=type)

        # Also save parameters
        parameters_dict = {self.device_name: {}}
        # Global
        for parameter in ds.get_parameter_list(device=self.device_name):
            parameters_dict[self.device_name][parameter] = str(ds.get_parameter(device=self.device_name, name=parameter))
        # Regional
        for region in ds.get_region_list(device=self.device_name):
            parameters_dict.update({region: {}})
            for parameter in ds.get_parameter_list(device=self.device_name, region=region):
                parameters_dict[region][parameter] = str(ds.get_parameter(device=self.device_name, region=region, name=parameter))
        with open(file.with_suffix(".yaml"), 'w') as yaml_file:
            yaml.dump(parameters_dict, yaml_file, default_flow_style=False)


    def ramp_dc_bias(self,
                ramp_parameters: ramp.RampParameters | None = None
                ):
        """
        Ramps the bias on "contact_name" to end_bias, changing step size as required to ensure convergence.

        Arguments:
            contact_name: Name of the contact to ramp (values in self._port_names_to_contact_names, or component port_name if not set)
            biases: The target biases to reach
            save_intermediate_structures_root: if not None, will save the fields under generated filenames
            initial_step_size: The initial step size for bias increase. Defaults to full range (end_bias - start_bias)
            maximum_step_size: The maximum step size allowed. Defaults to full range (end_bias - start_bias)
            step_size_scaling: how much to scale the step down when the ramp step fails to converge (and back up when it succeeds)
            min_step: The minimum step size allowed. Ramp will abort if step is reduced below this value.
            max_iter: The maximum solver number of iterations to attempt reaching the end_bias
            rel_error: The relative solver error tolerance
            abs_error: The absolute solver error tolerance
        """
        # Unpack ramp parameters
        if ramp_parameters is None:
            raise ValueError("ramp_parameters must be set")
        contact_name = ramp_parameters.contact_name
        biases = ramp_parameters.biases
        save_intermediate_structures_root = ramp_parameters.save_intermediate_structures_root
        initial_step_size = ramp_parameters.initial_step_size
        maximum_step_size = ramp_parameters.maximum_step_size
        step_size_scaling_down = ramp_parameters.step_size_scaling_down
        step_size_scaling_up = ramp_parameters.step_size_scaling_up
        min_step = ramp_parameters.min_step
        max_iter = ramp_parameters.max_iter
        rel_error = ramp_parameters.rel_error
        abs_error = ramp_parameters.abs_error
        ramp_parameters.clear_intermediate_structures_filepath()

        # Get contact mapping
        simulation_parameters = self.contact_parameters_from_contact(contact=contact_name)
        contact_region = self.simulation_inputs[2][contact_name]

        total_currents = []
        for end_bias in biases:
            start_bias = float(ds.get_parameter(device=self.device_name, name=simulation_parameters[0]))
            print(f"Ramping contact {contact_name} on region {contact_region} from V={start_bias:1.3f} V to V={end_bias:1.3f}")

            # Default step size
            initial_step_size = initial_step_size or abs(end_bias - start_bias)
            maximum_step_size = maximum_step_size or abs(end_bias - start_bias)

            ramp.rampbias(self.device_name,
                        simulation_parameters,
                        start_bias,
                        end_bias,
                        initial_step_size,
                        maximum_step_size,
                        step_size_scaling_down,
                        step_size_scaling_up,
                        min_step,
                        max_iter,
                        rel_error,
                        abs_error,
                    )
            current_total_currents = {}
            for contact in self._port_names_to_contact_names.values():
                current_total_currents.update({f"V___{contact}": self.get_voltage(contact=contact),
                                                    f"Ie___{contact}": self.get_current(contact=contact, carrier="electron"),
                                                    f"Ih___{contact}": self.get_current(contact=contact, carrier="hole"),
                                                    f"I___{contact}": self.get_current(contact=contact, carrier="total"),
                                                    })
            total_currents.append(current_total_currents)
            if save_intermediate_structures_root is not None:
                filepath = self.save_directory / save_intermediate_structures_root
                contact_str = ""
                for contact in self._port_names_to_contact_names.values():
                    contact_str += f"___{contact}_{self.get_voltage(contact=contact):1.3f}".replace(".", "p")
                filepath = filepath.with_name(filepath.stem + contact_str + ".ds")
                ramp_parameters.add_intermediate_structures_filepath(filepath)
                self.write(file=filepath, type="devsim")
                filepath = filepath.with_suffix(".dat")
                self.write(file=filepath, type="tecplot")

        return pd.DataFrame(total_currents)

    def get_voltage(self, contact: str) -> float:
        """Read voltage on contact."""
        simulation_contacts = self.contact_parameters_from_contact(contact=contact)
        return float(ds.get_parameter(device=self.device_name, name=simulation_contacts[0]))


    def get_charge(self, contact: str) -> float:
        """Get charge on a contact region.

        Arguments:
            contact: contact name
        """
        simulation_contacts = self.contact_interfaces_from_contact(contact=contact)
        charge = 0
        for simulation_contact in simulation_contacts:
            if "PotentialEquation" in ds.get_contact_equation_list(device=self.device_name, contact=simulation_contact):
                charge += ds.get_contact_charge(
                    device=self.device_name, contact=simulation_contact, equation="PotentialEquation"
                )
        return charge


    def get_current(self, contact: str, carrier: Literal["electron", "hole", "total"] = "total") -> float:
        """Accumulate charge on a contact region.

        Arguments:
            contact: contact name
            carrier: current type (electron, hole, or total)
        """
        simulation_contacts = self.contact_interfaces_from_contact(contact=contact)
        electron_current = 0
        hole_current = 0
        for simulation_contact in simulation_contacts:
            if "ElectronContinuityEquation" in ds.get_contact_equation_list(device=self.device_name, contact=simulation_contact):
                electron_current += ds.get_contact_current(
                    device=self.device_name, contact=simulation_contact, equation="ElectronContinuityEquation"
                )
            if "HoleContinuityEquation" in ds.get_contact_equation_list(device=self.device_name, contact=simulation_contact):
                hole_current = ds.get_contact_current(
                    device=self.device_name, contact=simulation_contact, equation="HoleContinuityEquation"
                )
        if carrier == "electron":
            return electron_current
        elif carrier == "hole":
            return hole_current
        elif carrier == "total":
            return electron_current + hole_current

    def set_parameter(self,
                      name: str,
                      value: float,
                      device: str | None = None,
                      region: str | None = None,
                    ):
        """Wrapper around DEVSIM set_parameter."""
        device = device or self.device_name
        if region is None:
            for region in ds.get_region_list(device=device):
                ds.set_parameter(device=device, region=region, name=name, value=value)
        else:
            ds.set_parameter(device=device, region=region, name=name, value=value)


    def get_node_field_values(self,
                  field: str = "NetDoping",
                  regions: Tuple[str] | None = None,
                  default_val = np.nan,
                  ):
        """Returns field data, in node index order.

        field: name of the field to show
        regions: name of regions to include. Defaults to all.
        default_val: value to assign if the field is not present in the region. Default Nan.
        """
        node_model_values = []
        regions = regions or ds.get_region_list(device=self.device_name)
        for region in regions:
            if field not in ds.get_node_model_list(device=self.device_name, region=region):
                node_model_values.extend(len(ds.get_node_model_values(device=self.device_name, region=region, name='x')) * [default_val])
            else:
                node_model_values.extend(ds.get_node_model_values(device=self.device_name, region=region, name=field))
        return np.array(node_model_values)


    def probe_field_values(self,
                  positions: Tuple[Tuple[float,float]] | Tuple[Tuple[float,float,float]],
                  interpolation_type: Literal["nearest", "linear", "cubic"] = "linear",
                  field: str = "NetDoping",
                  ):
        """Returns an interpolated field value at a specific location.

        Arguments:
            positions: list of (x,y) or (x,y,z) positions to probe
            interpolation_type: type of interpolation
            field: field to probe
            regions: regions to consider
        field: name of the field to show
        regions: name of regions to include. Defaults to all.
        """
        # if len(positions[0]) == 3:
        #     raise NotImplementedError("3D is not supported yet!")
        regions = []
        for region in ds.get_region_list(device=self.device_name):
            if field in ds.get_node_model_list(device=self.device_name, region=region):
                regions.append(region)
        x = self.get_node_field_values(field="x",
                                        regions=regions,
                                        )
        y = self.get_node_field_values(field="y",
                                        regions=regions)
        values = self.get_node_field_values(field=field,
                                        regions=regions)
        points = np.column_stack((x, y))
        return griddata(points, values, np.array(positions), method=interpolation_type)


    def plot2D(self,
               field: str | None = None,
               field_operation: Callable | None = identity,
               cmap: str | None = None,
               window_size: Tuple[int, int] = (1200, 1000),
               show_wireframe: bool = True,
               wireframe_color: Tuple[int, int, int] = (0,0,0),
               wireframe_transparency: float = 0.5,
               line_width: float = 0.1,
               log_scale: bool = False,
               jupyter_backend: str | None = 'trame',
               file: Path | None = None,
               title: str | None = None,
               # Camera options
               zoom: float = 1.0,
               camera_dx: float = 0.0,
               camera_dy: float = 0.0,
               position_x_scalar_bar: float = 0.1,
               position_y_scalar_bar: float = 0.1,
               n_labels_scalar_bar: int | None = None,
               title_scalar_bar: str | None = None
               ):
        """Plot the field using pyvista

        Arguments:
            field (str): The name of the field to be plotted. If None, plots the mesh with physical domains.
            field_operation (callable): transformation to apply to the data before plotting.
            cmap (str): The colormap to be used for the plot. Defaults to "jet".
            window_size (Tuple[int, int]): The size of the window in which the plot will be displayed. Defaults to (1200, 1000).
            show_edges (bool): Whether to show the edges of the mesh in the plot. Defaults to True.
            line_width (float): The width of the lines used to show edges
            edge_color (str): color of the mesh edges
            log_scale (bool): Whether to display the field values on a logarithmic scale. Defaults to False.
            jupyter_backend (str | None): The backend to be used when integrating with Jupyter notebooks. If None, integration is disabled. Defaults to 'html'.
            file: data file to load for plotting. Defaults to self.mesh_filename if field = None, or self.device_data_filename if a field is given.
        """
        # Parse backend
        if jupyter_backend is not None:
            pv.set_jupyter_backend(jupyter_backend)
            notebook = True
        else:
            notebook = False

        if field is None:
            mesh = meshio.read(file or self.save_directory / self.mesh_filename, file_format="gmsh")
            annotations = {value[0]: key for key, value in mesh.field_data.items()}
            mesh = pv.wrap(mesh)
            field = "gmsh:physical"
            title_scalar_bar = "Regions and interfaces"
            n_labels_scalar_bar = 0
            if cmap is None:
                cmap = get_distinguishable_colors(register=True)
        else: # go through tecplot for now (fix later)
            temp_path = file or self.save_directory / self.device_data_filename
            self.write(file=temp_path.with_suffix(".dat"), type="tecplot")
            reader = pv.get_reader(temp_path)
            mesh = reader.read()
            for block in mesh:
                block[field] = field_operation(block[field])
            annotations = {}
            if cmap is None:
                cmap = "jet"

        # Plot
        plotter = pv.Plotter(window_size=window_size, notebook=notebook)
        plotter.add_mesh(mesh,
                        scalars=field,
                        cmap=cmap,
                        show_edges=False,
                        log_scale=log_scale,
                        annotations=annotations
                        )
        if show_wireframe:
            wireframe_color = list(wireframe_color) + [wireframe_transparency]
            plotter.add_mesh(mesh,
                             style='wireframe',
                            color=wireframe_color,
                            line_width=line_width
                            )
        plotter.view_xy()
        plotter.remove_scalar_bar()
        plotter.add_scalar_bar(title=title_scalar_bar or field,
                                position_x=position_x_scalar_bar,
                                position_y=position_y_scalar_bar,
                                width=0.8,
                                height=0.05,
                                n_labels=n_labels_scalar_bar or 5,
                                )
        (x, y, z), (center_x, center_y, center_z), (up_x, up_y, up_z) = plotter.camera_position
        plotter.camera_position = (x - camera_dx, y - camera_dy, z), (center_x - camera_dx, center_y - camera_dy, center_z), (up_x, up_y, up_z)
        plotter.camera.zoom(zoom)  # Zoom in the camera by a factor of 1.2
        if title is not None and jupyter_backend is not None:
            plotter.add_text(title, position='upper_edge', font_size=20)
        elif title is not None:
            plotter.show(title=title)
        plotter.show()


    def get_parameter(self, parameter: str, region: str):
        return ds.get_parameter(device=self.device_name, region=region, name=parameter)



    def get_generated_parameters(self):
        """Returns DEVSIM parameters generated during the course of simulation. Used for restarting simulations with new meshes.

        All other parameters can be reloaded from input files (self.initialize again with same inputs).

        As implemented, this is all the global parameters.
        """
        return {parameter: ds.get_parameter(device=self.device_name, name=parameter)  for parameter in ds.get_parameter_list(device=self.device_name)}


    def remesh(self,
               remeshings: Dict,
               contact_biases: Dict | None = None,
               default_mincl: float = 0.0005,
               default_maxcl: float = 1.0,
               max_iterations: int = 10,
               threads_available: int = 10,
               solver_absolute_error: float = 1E10,
                solver_relative_error: float = 1E-10,
                solver_maximum_iterations: int = 100,
                solve_flag: bool = False,
                save_intermediate_data_root: str | None = None,
                save_intermediate_mesh_root: str | None = None,
                extended_solver: bool = True,
                extended_model: bool = True,
                extended_equation: bool = True,
                threads_task_size=2048,
                # Ramp solve settings
                ramp_parameters: ramp.RampParameters | None = None,
               ):
        """
        This method is responsible for remeshing the simulation structure based on the provided parameters.
        It utilizes the refinement.remesh_structure function to perform the remeshing process.

        TODO: this was brought from a standalone function in gdevsim.meshing.refinement; refactoring to use class attributes could be better.

        Arguments:
            - remeshings (dict): Dict with keys regions, and values list of RemeshingStrategy
            - default_mincl (float): The default minimum characteristic length for the mesh elements.
            - default_maxcl (float): The default maximum characteristic length for the mesh elements.
            - max_iterations (int): The maximum number of iterations to perform for the remeshing process.
            - threads_available (int): The number of threads available for parallel processing during remeshing.
            - solver_absolute_error (float): The absolute error tolerance for the solver used in the remeshing process.
            - solver_relative_error (float): The relative error tolerance for the solver used in the remeshing process.
            - solver_maximum_iterations (int): The maximum number of iterations for the solver used in the remeshing process.
            - solve_flag (bool): A flag indicating whether to solve the simulation after remeshing.
            - save_intermediate_data_root (str): The root directory for saving intermediate data during the remeshing process.
            - save_intermediate_mesh_root (str): The root directory for saving intermediate mesh files during the remeshing process.

        Returns:
            - steps: list of step indices
            - bisections: Dict with keys region, and values list of number of bisections at each step

        """
        from gdevsim.meshing import refinement


        steps, bisections = refinement.remesh_structure(
            restart_parameters=self.get_generated_parameters(),
            remeshings=remeshings,
            device_settings_filepath=self.save_directory / self.settings_filename,
            save_intermediate_data_root=save_intermediate_data_root,
            save_intermediate_mesh_root=save_intermediate_mesh_root,
            final_mesh_filepath=self.save_directory / self.mesh_filename,
            default_mincl=default_mincl,
            default_maxcl=default_maxcl,
            max_iterations=max_iterations,
            threads_available=threads_available,
            solver_absolute_error=solver_absolute_error,
            solver_relative_error=solver_relative_error,
            solver_maximum_iterations=solver_maximum_iterations,
            solve_flag=solve_flag
        )

        return steps, bisections


def initialize(
    # CAD settings
    component,
    layer_stack,
    mesh_type: str = "uz",
    xsection_bounds: Tuple[Tuple[float, float], Tuple[float, float]] | None = None,
    z: float | None = None,
    # Meshing settings
    resolutions: dict | None = None,
    remeshing_file: Path | None = None,
    global_scaling: float = 1e-4,  # gdsfactory is in um, semiconductor physics in cm
    default_characteristic_length=1.0,
    wafer_padding=2,
    wafer_layer=(99999, 0),
    interface_delimiter: str = "___",
    contact_delimiter: str = "@",
    threads_available: int  = 18,
    # Save settings
    reset_save_directory: bool = False,
    append_to_log: bool = False,
    print_log: bool = True,
    # Constants and settings for models (dict from yaml files for now)
    materials_parameters: Dict = get_all_materials(),
    global_parameters: Dict = get_global_parameters(),
    physics_parameters: Dict[str, Tuple[str]] = get_default_physics(),
    # Solver settings
    device_name: str | None = None,
    device_mesh_name: str | None = None,
    # Save files
    save_directory: Path | str = None,
    log_filename: str = "simulation.log",
    error_filename: str = "simulation.err",
    mesh_filename: str = "mesh_initial.msh2",
    settings_filename: str = "settings.pkl",
    simulation_gds_filename: str = "device.gds",
    device_data_filename: str = "device.dat",
):
    """
    Initilize a DEVSIM simulation object from the provided information.

    Args:
        ====== CAD settings ======

        component: The component to simulate. Needs to have net layers (with contact delimiter).
        layer_stack: The updated layerstack. Needs to contain the net layers.

        ====== Meshing settings ======

        mesh_type: The type of mesh to be used: 2D cross-section "uz", 2D in-plane "xy", or "3D".
        xsection_bounds: If "uz", the bounds of the cross section.
        z: if "xy", the plane of the cross-section.
        resolutions: The resolutions dict for the simulation. Useful for initial meshing.
        remeshing_file: .pos file to set resolution (overrides resolutions if provided).
        global_scaling: The unit conversion between gdsfactory (um) and semiconducor physics units (cm). Default is 1e-4. Change only if you know what you are doing.
        default_characteristic_length: The default characteristic length of the mesh. Default is 1.0 (um).
        wafer_padding: The padding in um for the wafer layer added to the component. Default is 2.
        interface_delimiter: The delimiter for mesh interface physicals. Default is "___".
        contact_delimiter: The delimiter for the contact physicals. Default is "@".

        ====== Model settings ======

        materials_parameters: Dict containing all materials parameters.
        global_parameters: Dict containing all global parameters.
        optical_generation_profiles: Dict[str, Dict] | None = None,
        field_initialization_override: Dict of dicts: {region: {field_name, (x,y,z,field_data)}}, that, if not None, is used to assign new initial values to the field. Uses unstructured interpolation.

        ====== DEVSIM settings ======

        device_name: The name of the device. Default is "temp_device".
        extended_solver: Whether to use the extended solver. Default is True.
        extended_model: Whether to use the extended model. Default is True.
        extended_equation: Whether to use the extended equation. Default is True.
        threads_available: The number of threads available. Default is 18.
        threads_task_size: The task size for the threads. Default is 1024.
        absolute_error: The absolute error for the initial solve. Default is 1e10.
        relative_error: The relative error for the initial solve. Default is 1e-12.
        maximum_iterations: The maximum number of iterations for the initial solve. Default is 100.

        ====== Save settings ======

        save_directory: directory where all output is saved. Defaults to PATH.simulation.{hash}
        reset_save_directory: if True, deletes previous save_directories of the same name
        log_filename: filename for logging (stdout). Defaults to simulation.log
        append_to_log: if True, appends to an existing log instead of creating a new one
        mesh_filename: filename for the generated mesh. Defaults to mesh_initial.msh2
        preinitialization_device_filename: The filepath for the device before initial solve. Useful to inspect simulation if solve crashes. Defaults to mesh_initial.dat (tecplot files).
        initialized_device_filepath: The filepath for the final structure. Useful to inspect the initialized simulation.
        settings_filename: The filepath for the device settings

    Returns:
        final_structure_filepath
    """
    # Save settings
    settings = locals().copy()

    if mesh_type == "3D":
        warnings.warn("3D simulation may suffer from numerical issues, use at your own risk!", UserWarning)

    # Create filepaths
    save_directory = Path(save_directory)
    if save_directory.exists() and reset_save_directory:
        shutil.rmtree(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)
    settings["save_directory"] = save_directory  # update if the argument was None
    log_filepath = save_directory / log_filename
    error_filepath = save_directory / error_filename
    mesh_filepath = save_directory / mesh_filename
    settings_filepath = save_directory / settings_filename
    simulation_gds_filepath = save_directory / simulation_gds_filename
    device_data_filepath = save_directory / device_data_filename

    log_file = open(log_filepath, "a") if append_to_log else open(log_filepath, "w")
    error_file = open(error_filepath, 'a') if append_to_log else open(error_filepath, "w")
    if print_log:
        sys.stdout = Tee(sys.stdout, log_file)
        sys.stderr = Tee(sys.stderr, error_file)
    else:
        sys.stdout = log_file
        sys.stderr = error_file

    # Print simulation information header
    if append_to_log is False:
        print("===============================================================")
        print(f"Simulation device: {device_name}")
        print("---------------------------------------------------------------")
        print("Component:")
        print("---------------------------------------------------------------")
        print(yaml.dump(component.to_dict()))
        print("---------------------------------------------------------------")
        print("LayerStack:")
        print("---------------------------------------------------------------")
        print(yaml.dump(layer_stack.to_dict()))
        print("---------------------------------------------------------------")
        print("Settings:")
        print("---------------------------------------------------------------")
        print(
            yaml.dump(
                {
                    k: v
                    for k, v in settings.items()
                    if k not in ["component", "layer_stack"]
                }
            )
        )
        print("===============================================================")

    # Separate out the layer stack into geometrical and doping layers
    layer_stack_geometry = layer_stack.filtered(
        [k for k, v in layer_stack.layers.items() if v.layer_type != "doping"]
    )
    layer_stack_dopings = layer_stack.filtered(
        [k for k, v in layer_stack.layers.items() if v.layer_type == "doping"]
    )

    # Generate initial mesh
    print("===============================================================")
    meshing_string = (
        "resolutions dict" if remeshing_file is None else str(remeshing_file)
    )
    print(f"Meshing from {meshing_string}")
    print("===============================================================")
    get_mesh(
        component=component,
        layer_stack=layer_stack_geometry,
        type=mesh_type,
        xsection_bounds=xsection_bounds,
        z=z,
        wafer_padding=wafer_padding,
        wafer_layer=wafer_layer,
        resolutions=resolutions,
        default_characteristic_length=default_characteristic_length,
        filename=str(mesh_filepath),
        interface_delimiter=interface_delimiter,
        global_scaling=global_scaling,
        background_remeshing_file=remeshing_file,
        n_threads=threads_available,
        # progress_bars=False
    )

    # Parse mesh information for DEVSIM
    mesh = meshio.read(str(mesh_filepath), file_format="gmsh")
    if mesh_type == "uz" or mesh_type == "xy":
        dimension = 2
    elif mesh_type == "3D":
        dimension = 3
    interfaces = {
        key for key, value in mesh.field_data.items() if value[1] == dimension - 1
    }
    regions = {key for key, value in mesh.field_data.items() if value[1] == dimension}

    # Contacts are the interfaces with the contact delimiter
    contact_interfaces = {
        interface for interface in interfaces if contact_delimiter in interface
    }
    interfaces -= contact_interfaces

    # Parse materials from layer stack
    materials_dict = {
        region: layer_stack_geometry.layers[region].material for region in regions
    }
    materials_dict["None"] = {"type": None}

    # Sort the regions again
    regions_priority = sorted(
        regions, key=lambda x: layer_stack_geometry.layers[x].mesh_order
    ) + ["None"]

    # Define DEVSIM simulation
    print("===============================================================")
    print("Performing GMSH --> DEVSIM mesh conversion")
    print("===============================================================")
    devsim_device_from_mesh(
        regions=regions,
        interfaces=interfaces,
        contact_interfaces=contact_interfaces,
        materials_dict=materials_dict,
        interface_delimiter=interface_delimiter,
        mesh_name=device_mesh_name,
        mesh_filepath=str(mesh_filepath),
        device_name=device_name,
        reset=True,
        regions_priority=regions_priority,
        dimension=dimension,
    )

    # Create material indices
    for region in regions:
        models.create_material_indexing(
            device=device_name, region=region, interface_delimiter=interface_delimiter
        )

    # Assign initial models

    print("===============================================================")
    print("Assigning models")
    print("===============================================================")

    # NOTE: Dopings are special because they are cumulative, for TCAD we can always reasonably expect to have them, and they are tied to the GDS/LayerStack (unlike other fields like optical generation)
    # But we could also still do this outside of this function, and treat them as regular "extra fields"
    # We could also use DEVSIM itself to add the profiles intead of doing it upstream (but would only work for registered functions in SYMDIFF)

    # Extract doping profiles from component and layer stack
    if mesh_type == "uz":
        # Parse the LayerStack, Component, and bounds to extract doping bounds
        doping_data = parse_layer_stack_doping_uz(
            layer_stack_dopings, layer_stack_geometry, component, xsection_bounds
        )
        # Accumulate dopants across different layers
        total_profiles = project_profiles_uz(layer_stack_geometry, doping_data)
        # Introduce the dopant information to DEVSIM
        assign_doping_profiles_uz(
            regions,
            materials_dict,
            materials_parameters,
            total_profiles,
            layer_stack_geometry,
            device_name,
            global_scaling,
        )
    elif mesh_type == "xy":
        # Parse the LayerStack, Component, and bounds to extract doping bounds
        doping_data = parse_layer_stack_doping_xy(
            layer_stack_dopings, component
        )
        # Introduce the dopant information to DEVSIM
        assign_doping_profiles_xy(
            regions,
            materials_dict,
            materials_parameters,
            doping_data,
            layer_stack_geometry,
            device_name,
            global_scaling,
        )


    # Setup models according to materials_parameters
    print("Assigning region models")
    print("---------------------------------------------------------------")

    opts = {}
    for region in regions:
        opts.update(
            assign_region_models(
                region,
                materials_parameters,
                materials_dict,
                global_parameters,
                physics_parameters,
                device_name,
                interface_delimiter=interface_delimiter,
            )
        )

    print("---------------------------------------------------------------")
    print("Assigning interface models")
    print("---------------------------------------------------------------")

    for interface in interfaces:
        assign_interface_models(
            device_name, interface, materials_parameters, materials_dict
        )

    print("---------------------------------------------------------------")
    print("Assigning contact models")
    print("---------------------------------------------------------------")

    for contact_interface in contact_interfaces:
        assign_contact_models(
            contact_interface,
            device_name,
            materials_parameters,
            materials_dict,
            interface_delimiter,
            contact_delimiter,
            opts,
        )

    # Save settings
    settings["component"] = simulation_gds_filepath
    with open(settings_filepath, "wb") as file:
        dill.dump(settings, file)
    component.write_gds(simulation_gds_filepath, with_metadata=True)

    return settings_filepath



def override_field_values(device_name, field_initialization_override_dict: Dict):
    """Override field values of a device using unstructured interpolation from a field_initialization_override_dict:

    Arguments:
        field_initialization_override_dict: dict with keys: region_names, and values dict:
            x: x values
            y: y values
            z: z values
            field_name: (field_data, post_interpolation_transformation)
    """
    for region_name, field_data_dict in field_initialization_override_dict.items():
        for field_name in field_data_dict:
            if field_name == "x" or field_name == "y" or field_name == "z":
                continue
            else:
                field_data, val_transformation = field_data_dict[field_name]
                add_unstructured_data_to_mesh(device = device_name,
                                                region = region_name,
                                                name = field_name,
                                                x_array = field_data_dict["x"],
                                                y_array = field_data_dict["y"],
                                                z_array = field_data_dict["z"],
                                                val_array= field_data,
                                                val_transformation = val_transformation,
                                                )


def solve(
    save_directory: Path | None = None,
    device_data_filename: Path | None = None,
    solve_type: Literal['dc', 'ac', 'noise', 'transient_dc', 'transient_bdf1', 'transient_bdf2', 'transient_tr'] = 'dc',
    extended_solver: bool = True,
    extended_model: bool = True,
    extended_equation: bool = True,
    threads_available: int = 18,
    threads_task_size: int = 2048,
    absolute_error: float = 1e10,
    relative_error: float = 1e-8,
    charge_error: float = 0.0,
    maximum_iterations=100,
    ):

    device_data_filepath = save_directory / device_data_filename

    # Solver settings
    set_parameter(name="direct_solver", value="mkl_pardiso")
    set_parameter(name="extended_solver", value=extended_solver)
    set_parameter(name="extended_model", value=extended_model)
    set_parameter(name="extended_equation", value=extended_equation)
    set_parameter(name="threads_available", value=threads_available)
    set_parameter(name="threads_task_size", value=threads_task_size)

    # Initialization solve
    ds.solve(
        type=solve_type,
        absolute_error=absolute_error,
        relative_error=relative_error,
        maximum_iterations=maximum_iterations,
    )

    return device_data_filepath



def optical_generation_from_file(
    device,
    region,
    genrate=1,
    gen_npz_file="generation.npy",  # from 0 to 1 (profile)
    x_npz_file="x_meep.npy",
    y_npz_file="y_meep.npy",
    xy_offset=(0, 0),
    global_scaling: float = 1e-4,
) -> None:
    """Set values of node model OptGen to add optical generation in region."""

    # Load generation data
    xyz_gen_npz = np.load(gen_npz_file)
    x_npz = (np.load(x_npz_file) - xy_offset[0]) * global_scaling
    y_npz = (np.load(y_npz_file) - xy_offset[0]) * global_scaling

    # Add acceptors
    add_structured_data_to_mesh(
        device=device,
        region=region,
        name="OptGen",
        x_array=x_npz,  # match units
        y_array=y_npz,  # match units
        z_array=None,
        val_array=xyz_gen_npz * genrate,
    )


if __name__ == "__main__":
    from gdsfactory.components import straight_pn, via_stack
    from gdsfactory.components.via import viac
    from gdsfactory.typings import ComponentSpec

    from gdevsim.samples.layers_photonic import LAYER, get_layer_stack_photonic

    component = straight_pn(
        length=15,
        via_stack=gf.partial(
            via_stack,
            layers=(
                None,
                LAYER.M1,
            ),
            vias=(viac,),
        ),
        via_stack_width=2,
        taper=None,
    )

    @gf.cell
    def straight_pn_via_ports(component: ComponentSpec):
        """Process the component to define ports on the VIAC layer."""
        c = gf.Component()

        mod_ref = c << component

        # Get top and bottom vias location
        vias = component.extract(layers=[LAYER.VIAC])
        top_via_y = vias.ymax
        bot_via_y = vias.ymin

        # Define ports there
        c.add_port(
            name="e_p",
            center=(mod_ref.xsize / 2, top_via_y - 0.01),
            layer=LAYER.VIAC,
            width=vias.xsize,
            orientation=90,
        )
        c.add_port(
            name="e_n",
            center=(mod_ref.xsize / 2, bot_via_y + 0.01),
            layer=LAYER.VIAC,
            width=vias.xsize,
            orientation=270,
        )

        return c

    component = straight_pn_via_ports(component)
    layer_stack = get_layer_stack_photonic()

    component.show()

    resolutions = {
        "slab": {"resolution": 0.05, "distance": 1.0},
        "core": {"resolution": 0.05, "distance": 1.0},
        "clad": {"resolution": 0.5, "distance": 1.0},
        "box": {"resolution": 0.5, "distance": 1.0},
    }

    xsection_bounds = ((15 / 2, -5), (15 / 2, 5))

    simulation = DevsimComponent(
        component=component,
        layer_stack=layer_stack,
        mesh_type="uz",
        xsection_bounds=xsection_bounds,
        _port_names_to_contact_names={"e_p": "anode", "e_n": "cathode"},
        resolutions=resolutions,
    )

    physics = {
        "silicon": {
            "mobility": ("constant",),
            "generation_recombination": ("bulkSRH",),
        },
        "germanium": {
            "mobility": ("constant",),
            "generation_recombination": ("bulkSRH",),
        },
        "aluminum": {},
        "silicon_dioxide": {},
    }

    output_filename_settings = simulation.initialize(
        reset_save_directory=True,
        threads_available=12,
        physics_parameters=physics,
    )

    # Apply remeshing
    from gdevsim.meshing import refinement

    # steps, bisections = refinement.remesh_structure(
    #     remeshings=refinement.default_remeshing_presolve,
    #     device_settings_filepath=output_filename_settings,
    #     save_intermediate_structures=True,
    #     default_mincl=0.0005,  # 10 nm
    #     default_maxcl=1,  # 1 um
    #     max_iterations=5,
    #     threads_available=6,
    #     # solver_absolute_error=1e12,
    #     # solver_relative_error=1e-9,
    #     # solver_maximum_iterations=100,
    # )
    # import matplotlib.pyplot as plt
    # for region in bisections.keys():
    #     plt.plot(steps, bisections[region], label=region)
    # plt.xlabel("Remeshing step")
    # plt.ylabel("Bisections")
    # plt.legend(title="Region")
    # # plt.show()

    steps0, bisections0 = refinement.remesh_structure(
        remeshings=refinement.default_remeshing_presolve,
        device_settings_filepath=output_filename_settings,
        save_intermediate_structures_root="device_remeshed_doping",
        default_mincl=0.0005,  # 10 nm
        default_maxcl=1,  # 1 um
        max_iterations=1,
        solve_flag=False,
        threads_available=12,
    )

    simulation.solve(
        threads_available=6,
        absolute_error=1e12,
        relative_error=1e-6,
    )

    steps1, bisections1 = refinement.remesh_structure(
        remeshings=refinement.default_remeshing_postsolve,
        device_settings_filepath=output_filename_settings,
        save_intermediate_structures_root="device_remeshed_solve_lowres",
        default_mincl=0.0005,  # 10 nm
        default_maxcl=1,  # 1 um
        max_iterations=1,
        solve_flag=True,
        threads_available=12,
        solver_absolute_error=1e12,
        solver_relative_error=1e-6,
        solver_maximum_iterations=100,
    )

    steps2, bisections2 = refinement.remesh_structure(
        remeshings=refinement.default_remeshing_postsolve,
        device_settings_filepath=output_filename_settings,
        save_intermediate_structures_root="device_remeshed_solve_highres",
        default_mincl=0.0005,  # 10 nm
        default_maxcl=1,  # 1 um
        max_iterations=1,
        solve_flag=True,
        threads_available=12,
        solver_absolute_error=1e12,
        solver_relative_error=1e-12,
        solver_maximum_iterations=100,
    )

    import matplotlib.pyplot as plt

    for region in bisections0.keys():
        plt.plot(steps0, bisections0[region], label=f"{region}_0")
        plt.plot(steps1, bisections1[region], label=f"{region}_1")
        plt.plot(steps2, bisections2[region], label=f"{region}_2")
    plt.xlabel("Remeshing step")
    plt.ylabel("Bisections")
    plt.legend(title="Region")
    plt.show()

    # total_doping_profiles = initialize_devsim_simulation(
    #     component,  # use component with net layers
    #     layer_stack,  # use updated layerstack with net layers
    #     resolutions="bgmesh.pos",
    #     xsection_bounds=xsection_bounds,
    #     default_characteristic_length=1.0,
    #     wafer_padding=2,
    #     mesh_filename="temp_remeshed.msh2",
    # )

    # for target_region in total_doping_profiles.keys():

    #     x_samplings = total_doping_profiles[target_region]["x_samplings"]
    #     y_samplings = total_doping_profiles[target_region]["y_samplings"]

    #     # Default values
    #     total_doping_profiles[target_region]["net"] = total_doping_profiles[target_region]["donor"] - total_doping_profiles[target_region]["acceptor"]

    #     for key in ["donor", "acceptor", "net"]:

    #         plt.pcolormesh(x_samplings * 1E-4, y_samplings * 1E-4, total_doping_profiles[target_region][key], shading="auto")
    #         plt.colorbar(label="Concentration (cm-3)")
    #         plt.xlabel('x')
    #         plt.ylabel('y')
    #         plt.title(f"Region: {target_region}, type: {key}")
    #         plt.show()
