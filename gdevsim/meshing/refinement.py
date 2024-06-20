"""Adapted from J. Sanchez https://github.com/devsim/devsim_misc/blob/main/refinement/refinement2.py

Improvements:
    - parametrize remeshing strategies
    - unify code (e.g. oxide + semiconductor use same functions)
    - use array operations (still some left to convert)
    - add coordinate scaling
    - return bisection count
    - remesh on model values without solving, or solutions after solving
    - initialize new solution from interpolated past solution
    - remeshing criteria log difference with threshold for automatic stop
    - cleanup code
"""
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil

from typing import Dict

import devsim as ds
import dill
import gdsfactory as gf
import numpy as np

from gdevsim.simulation import initialize, override_field_values, solve
from gdevsim.utils.operations import thresholded_log_difference, identity
from gdevsim import ramp

@dataclass
class RemeshingStrategy:
    field: str
    field_operation: Callable
    threshold: float
    post_interpolation_transformation: Callable = identity


default_remeshing_strategies_presolve = (
    RemeshingStrategy(field="NetDoping",
                      field_operation=lambda field, x0, x1: thresholded_log_difference(field, x0, x1),
                      threshold=1,
                      ),
    # RemeshingStrategy(field="OptGen",
    #                   field_operation=lambda field, x0, x1: thresholded_log_difference(field, x0, x1),
    #                   threshold=1,
    #                   ),
)


default_remeshing_strategies_postsolve = (
    RemeshingStrategy(field="Potential",
                      field_operation=lambda field, x0, x1: np.abs(field[x0] - field[x1]),
                      threshold=0.05,
                      ),
    RemeshingStrategy(field="Electrons",
                      field_operation=lambda field, x0, x1: thresholded_log_difference(field, x0, x1),
                      threshold=1.0,
                      post_interpolation_transformation=lambda x: np.clip(x, a_min=np.finfo(float).eps, a_max=np.inf),
                      ),
    RemeshingStrategy(field="Holes",
                      field_operation=lambda field, x0, x1: thresholded_log_difference(field, x0, x1),
                      threshold=1.0,
                      post_interpolation_transformation=lambda x: np.clip(x, a_min=np.finfo(float).eps, a_max=np.inf),
                      ),
)

def create_remeshing_dict(remeshings: list[RemeshingStrategy]):
    """Convert RemeshingStrategy list into dict."""
    remeshing_dict = {}
    for region_name, remeshing_strategy_list in remeshings.items():
        remeshing_subdict = {}
        remeshing_subdict["fields"] = []
        remeshing_subdict["field_operations"] = []
        remeshing_subdict["thresholds"] = []
        for remeshing in remeshing_strategy_list:
            remeshing_subdict["fields"].append(remeshing.field)
            remeshing_subdict["field_operations"].append(remeshing.field_operation)
            remeshing_subdict["thresholds"].append(remeshing.threshold)
        remeshing_dict[region_name] = remeshing_subdict

    return remeshing_dict


default_remeshing_presolve = {
    # Photonic PDK
    "core": default_remeshing_strategies_presolve,
    "slab": default_remeshing_strategies_presolve,
    "ge": default_remeshing_strategies_presolve,
    # Electronic PDK
    "bulk": default_remeshing_strategies_presolve,
}

default_remeshing_postsolve = {
    # Photonic PDK
    "core": default_remeshing_strategies_postsolve,
    "slab": default_remeshing_strategies_postsolve,
    "ge": default_remeshing_strategies_postsolve,
    # Electronic PDK
    "bulk": default_remeshing_strategies_postsolve,
}


def print_header(fh) -> None:
    """
    Write header for backround mesh view
    """
    fh.write('View "background mesh" {\n')


def print_footer(fh) -> None:
    """
    Write footer for backround mesh view
    """
    fh.write("};\n")


def get_edge_index(
    device: str,
    region: str,
) -> np.ndarray:
    """
    maps element edges to regular edges
    """
    # now iterate over the edges of the element
    if "eindex" not in ds.get_element_model_list(device=device, region=region):
        ds.element_model(
            device=device, region=region, name="eindex", equation="edge_index"
        )
    eindex = ds.get_element_model_values(device=device, region=region, name="eindex")
    eindex = np.array(eindex, dtype=int)
    return eindex


def get_node_index(
    device: str,
    region: str,
) -> np.ndarray:
    """
    maps head and tail nodes of from their edge index
    """
    # identify all edges that need to be bisected
    # ultimately translated to an element
    if "node_index@n0" not in ds.get_edge_model_list(device=device, region=region):
        ds.edge_from_node_model(node_model="node_index", device=device, region=region)
    nindex = np.array(
        [
            ds.get_edge_model_values(
                device=device, region=region, name="node_index@n0"
            ),
            ds.get_edge_model_values(
                device=device, region=region, name="node_index@n1"
            ),
        ],
        dtype=int,
    ).T
    return nindex


def calculate_clengths(
    device: str, region: str, model_values, coordinate_scaling: float
):
    """
    calculate the characteristic lengths for each edge by bisecting the edge length
    """
    clengths = (
        np.array(
            ds.get_edge_model_values(device=device, region=region, name="EdgeLength")
        )
        / coordinate_scaling
    )
    bisection_count = 0
    for i, v in enumerate(model_values):
        if v != 0:
            clengths[i] *= 0.5
            bisection_count += 1
    print(f"Region: {region}, Edge Bisections: {bisection_count}")
    return clengths, bisection_count


def get_output_elements3(device, nindex, eindex, clengths, number_nodes, mincl, maxcl):
    """
    gets the node indexes and the characterisic lengths for each element
    device : device we are operating on
    nindex : from get_node_index
    eindex : from get_edge_index
    clengths : from calculate_clengths
    number_nodes : number of nodes
    mincl : minimum characteristic length
    maxcl : maximum characteristic length
    """
    # set upper limit to maxcl
    node_map = [maxcl] * number_nodes
    # get node indexes for each edge
    for i, n in enumerate(nindex):
        # clip minimum value to mincl
        v = max(clengths[i], mincl)
        for ni in n:
            node_map[ni] = min(node_map[ni], v)

    dim = ds.get_dimension(device=device)
    if dim == 2:
        # 3 edges per triangle
        skip = 3
    elif dim == 3:
        # 6 edges per tetrahedron
        skip = 6
    else:
        raise RuntimeError("Unhandled dimension %d" % dim)

    # break into a per element basis
    outputelements = []
    for i in range(0, len(eindex), skip):
        ndict = {}
        # mapping of element edge into an edge index
        for j in eindex[i : i + skip]:
            # mapping of edge index into a node index
            for k in nindex[j]:
                if k not in ndict:
                    ndict[k] = node_map[k]
        outputelements.append(tuple(ndict.items()))
    return outputelements


def print_elements(fh, device, region, elements, coordinate_scaling):
    """
    print background mesh triangles
    """
    x = (
        np.array(ds.get_node_model_values(device=device, region=region, name="x"))
        / coordinate_scaling
    )
    y = (
        np.array(ds.get_node_model_values(device=device, region=region, name="y"))
        / coordinate_scaling
    )
    z = (
        np.array(ds.get_node_model_values(device=device, region=region, name="z"))
        / coordinate_scaling
    )

    dim = ds.get_dimension(device=device)
    if dim == 2:
        shape = "ST"
    elif dim == 3:
        shape = "SS"
    else:
        raise RuntimeError("Unhandled dimension %d" % dim)

    for e in elements:
        coords = [coord for n, v in e for coord in (x[n], y[n], z[n])]
        values = [v for n, v in e]
        coordstring = ", ".join([format(x, "1.15g") for x in coords])
        valuestring = ", ".join([format(x, "1.15g") for x in values])
        fh.write(f"{shape}({coordstring}) {{{valuestring}}};\n")


def refine_common(fh, device, region, model_values, mincl, maxcl, coordinate_scaling):
    """
    prints out the refined elements
    model_values : non-zero for edges to be bisected
    mincl : minimum characteristic length
    maxcl : maximum characteristic length
    """
    clengths, bisection_count = calculate_clengths(
        device=device,
        region=region,
        model_values=model_values,
        coordinate_scaling=coordinate_scaling,
    )

    eindex = get_edge_index(device, region)
    nindex = get_node_index(device, region)
    number_nodes = len(
        ds.get_node_model_values(device=device, region=region, name="node_index")
    )

    outputelements = get_output_elements3(
        device=device,
        nindex=nindex,
        eindex=eindex,
        clengths=clengths,
        number_nodes=number_nodes,
        mincl=mincl,
        maxcl=maxcl,
    )
    print_elements(
        fh=fh,
        device=device,
        region=region,
        elements=outputelements,
        coordinate_scaling=coordinate_scaling,
    )

    return bisection_count


def get_model_values(
    device: str,
    region: str,
    fields: list[str],
    field_operations: list[Callable],
    thresholds: list[float],
    ramp_parameters: ramp.RampParameters | None = None,
):
    """
    returns a model for refinement
    """
    # edge to node mapping (node0, node1)
    node_index = np.array(get_node_index(device=device, region=region))

    merge_lists = []

    # Regular case with no accumulation
    if ramp_parameters is None:
        for field, field_operation, threshold in zip(fields, field_operations, thresholds):
            if field in ds.get_node_model_list(device=device, region=region):
                field_values = np.array(
                    ds.get_node_model_values(device=device, region=region, name=field)
                )
                operation_result = field_operation(
                    field_values, node_index[:, 0], node_index[:, 1]
                )
                merge_lists.append(np.where(operation_result > threshold, 1, 0))

    # We collect refinements over multiple biases, accumulate over the different saved solutions
    else:
        ramp_files = ramp_parameters.get_intermediate_structures_filepaths()
        for ramp_file in ramp_files:
            ds.reset_devsim()
            ds.load_devices(file=ramp_file)
            for field, field_operation, threshold in zip(fields, field_operations, thresholds):
                field_values = np.array(
                    ds.get_node_model_values(device=device, region=region, name=field)
                )
                operation_result = field_operation(
                    field_values, node_index[:, 0], node_index[:, 1]
                )
                merge_lists.append(np.where(operation_result > threshold, 1, 0))

    test_model = np.maximum.reduce(merge_lists)

    return test_model


def refine_region(
    fh,
    device,
    region,
    fields,
    field_operations,
    thresholds,
    mincl,
    maxcl,
    coordinate_scaling,
    ramp_parameters: ramp.RampParameters | None = None,
):
    """
    refinement for semiconductor regions
    mincl : minimum characteristic length
    maxcl : maximum characteristic length
    """

    # Handle default case
    if fields is None:
        test_model = [0.0] * len(
            ds.get_edge_model_values(device=device, region=region, name="EdgeLength")
        )
    else:
        test_model = get_model_values(
            device=device,
            region=region,
            fields=fields,
            field_operations=field_operations,
            thresholds=thresholds,
            ramp_parameters=ramp_parameters,
        )

    return refine_common(
        fh=fh,
        device=device,
        region=region,
        model_values=test_model,
        mincl=mincl,
        maxcl=maxcl,
        coordinate_scaling=coordinate_scaling,
    )


def create_remeshing_file(
    device_name: str = "temp_device",
    remeshings: dict = default_remeshing_presolve,
    background_remeshing_field_filepath: str = "bgmesh.pos",
    default_mincl: float = 0.01,  # 10 nm
    default_maxcl: float = 1,  # 1 um
    coordinate_scaling: float = 1e-4,
    ramp_parameters: ramp.RampParameters | None = None,
):
    """Adaptively remesh device given target regions and remeshing strategies.

    Arguments:
        device: device name
        remeshings: dict with key: region type, and values a remeshing dict with entries:
            mincl: minimum characteristic length allowed in the region
            maxcl: maximum characteristic length allowed in the region
            fields: list of fields
            field_operations: list of callables function(x0, x1) from which the remeshing criteria is calculated
            thresholds: list of numbers. Mesh nodes where field_operation() > threshold will be bisected.
        background_remeshing_field_filepath: where to save the intermediary gmsh remeshing field
        coordinate_scaling: scaling between mesh input sizes and DEVSIM-reported sizes
    """

    # Check that all region are included in remeshing (add default no remeshing otherwise)
    # Also make sure to ignore remeshing entries not in regions
    device_regions = ds.get_region_list(device=device_name)
    current_remeshing_dict = {}
    for region in device_regions:
        if region in device_regions:
            if region in remeshings:
                current_remeshing_dict[region] = remeshings[region]
            else:
                current_remeshing_dict[region] = {
                    "mincl": default_mincl,
                    "maxcl": default_maxcl,
                    "fields": None,
                    "field_operations": None,
                    "thresholds": None,
                }

    bisection_counts = {}

    with open(background_remeshing_field_filepath, "w") as fh:
        print_header(fh)

        for region, remeshing_dict in current_remeshing_dict.items():
            # If there is no min/max lc, use default
            if "mincl" not in remeshing_dict:
                remeshing_dict["mincl"] = default_mincl
            if "maxcl" not in remeshing_dict:
                remeshing_dict["maxcl"] = default_maxcl

            bisection_count = refine_region(
                fh=fh,
                device=device_name,
                region=region,
                fields=remeshing_dict["fields"],
                field_operations=remeshing_dict["field_operations"],
                thresholds=remeshing_dict["thresholds"],
                mincl=remeshing_dict["mincl"],
                maxcl=remeshing_dict["maxcl"],
                coordinate_scaling=coordinate_scaling,
                ramp_parameters=ramp_parameters,
            )
            bisection_counts[region] = bisection_count

        print_footer(fh)

    return bisection_counts



def prepare_field_initialization_override_dict(device_name, remeshings):
    field_initialization_override_dict = {}
    for region_name, refinement_list in remeshings.items():
        if region_name in ds.get_region_list(device=device_name):
            field_initialization_override_dict[region_name] = {}
            field_initialization_override_dict[region_name]["z"] = None
            field_initialization_override_dict[region_name]["x"] = ds.get_node_model_values(
                                                                        device=device_name,
                                                                        region=region_name,
                                                                        name="x"
                                                                        )
            field_initialization_override_dict[region_name]["y"] = ds.get_node_model_values(
                                                                        device=device_name,
                                                                        region=region_name,
                                                                        name="y"
                                                                        )
            for remeshing in refinement_list:
                if remeshing.field in ds.get_node_model_list(device=device_name, region=region_name):
                    field_initialization_override_dict[region_name][remeshing.field] = (ds.get_node_model_values(device=device_name, region=region_name, name=remeshing.field), remeshing.post_interpolation_transformation)

    return field_initialization_override_dict



def remesh_structure(
    # Device biases
    restart_parameters: Dict | None = None,
    remeshings=default_remeshing_presolve,
    device_settings_filepath: Path = None,
    default_mincl: float = 0.0005,  # 0.5 nm
    default_maxcl: float = 1,  # 1 um
    max_iterations: int = 10,
    save_intermediate_data_root: str = "device_remeshed",
    save_intermediate_mesh_root: str = "device_remeshed",
    final_mesh_filepath: Path | None = None,
    threads_available: int = 6,
    # Solver settings
    solve_flag: bool = False,
    extended_solver: bool = True,
    extended_model: bool = True,
    extended_equation: bool = True,
    threads_task_size=2048,
    solver_absolute_error=1e10,
    solver_relative_error=1e-9,
    solver_maximum_iterations=100,
):
    """Remesh for some number of steps or until fully bisected, and return the number of bisections vs iteration.

    Note that we only need the path for device_settings because all other paths are logged there.

    Arguments:
        remeshings: (dict) Dictionary with list of remeshing strategies for each region
        default_mincl: (float) Default minimum characteristic length
        default_maxcl: (float) Default maximum characteristic length
        device_settings_filepath: (Path) Path to the device settings file
        max_iterations: (int) Maximum number of iterations for remeshing
        save_intermediate_data_root: (str) Root directory to save intermediate data
        save_intermediate_mesh_root: (str) Root directory to save intermediate mesh
        final_mesh_filepath: (Path | None) Path to the final mesh file
        threads_available: (int) Number of available threads
        ramp_contact_dict: (dict | None) Dictionary with kwargs for the ramp
        solve_flag: (bool) Flag to indicate if the solver should be used
        extended_solver: (bool) Flag to indicate if the solver is extended
        extended_model: (bool) Flag to indicate if the model is extended
        extended_equation: (bool) Flag to indicate if the equation is extended
        threads_task_size: (int) Size of the task for each thread
        solver_absolute_error: (float) Absolute error for the solver
        solver_relative_error: (float) Relative error for the solver
        solver_maximum_iterations: (int) Maximum number of iterations for the solver
    """
    with open(device_settings_filepath, "rb") as file:
        settings = dill.load(file)
        save_directory = Path(settings["save_directory"])
    settings["component"] = gf.import_gds(settings["component"])

    # Remeshing
    steps = []
    bisections = []
    for n in range(1, max_iterations + 1):
        # Get previous initial values
        field_initialization_override_dict = prepare_field_initialization_override_dict(settings["device_name"], remeshings)

        # Get remeshing
        bisections_n = create_remeshing_file(
            device_name=settings["device_name"],
            remeshings=create_remeshing_dict(remeshings),
            background_remeshing_field_filepath=save_directory
            / f"remeshing_bgmesh_{n}.pos",
            default_mincl=default_mincl,
            default_maxcl=default_maxcl,
        )
        steps.append(n)
        bisections.append(bisections_n)

        # Terminate early if there are no more bisections
        if all(value == 0 for value in bisections_n.values()):
            break
        # If not, compute the new structure
        else:
            # Reinitialize with new mesh
            settings["append_to_log"] = True
            settings["reset_save_directory"] = False
            settings["mesh_filename"] = f"{save_intermediate_mesh_root}_{n}.msh2"
            settings["remeshing_file"] = str(
                save_directory / f"remeshing_bgmesh_{n}.pos"
            )
            settings["threads_available"] = threads_available
            initialize(**settings)
            if restart_parameters is not None:
                for parameter_name, parameter_value in restart_parameters.items():
                    ds.set_parameter(device=settings["device_name"], name=parameter_name, value=parameter_value)
            if solve_flag:
                filename = settings['device_data_filename']
                filename = filename.rsplit('.', 1)
                filename = f"{filename[0]}_remeshed_{n}.{filename[1]}"

                # Initialize with last values for efficiency
                override_field_values(device_name=settings["device_name"],
                                    field_initialization_override_dict=field_initialization_override_dict)
                
                solve(save_directory = settings["save_directory"],
                        device_data_filename = filename,
                        extended_solver = extended_solver,
                        extended_model = extended_model,
                        extended_equation = extended_equation,
                        threads_available = threads_available,
                        threads_task_size = threads_task_size,
                        absolute_error = solver_absolute_error,
                        relative_error = solver_relative_error,
                        maximum_iterations = solver_maximum_iterations,
                    )
                    
                if save_intermediate_data_root:
                    remeshed_filepath = save_directory / f"{save_intermediate_data_root}_{n}.dat"
                    ds.write_devices(file=str(remeshed_filepath), type="tecplot")
                if final_mesh_filepath:
                    shutil.copy(Path(save_directory / settings['mesh_filename']), final_mesh_filepath)

    # Format bisections for easier plotting
    bisection_by_region = {}
    for region in bisections[0].keys():
        if region in remeshings.keys():
            bisection_by_region[region] = [x[region] for x in bisections]

    return steps, bisection_by_region


if __name__ == "__main__":
    # Load test device
    # ds.load_devices(file="test_device.ds")

    # Apply remeshing
    remesh_structure(
        remeshings=default_remeshing_presolve,
        initial_device_filepath="test_device.ds",
        save_intermediate_structures=True,
    )