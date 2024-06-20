from pathlib import Path

import devsim as ds
from gdevsim.contacts import parse_contact_interface
from gdevsim.meshing.parse_gmsh import get_fix_interfaces


def delete_interface_physicals(
    input_filepath,
    output_filepath,
    start_string="$PhysicalNames\n",
    end_string="$EndPhysicalNames\n",
    dimension=1,
):
    """Deletes interface physicals in a GMSH file for reprocessing with DEVSIM code."""

    with open(input_filepath) as file:
        lines = file.readlines()

    start_line = None
    end_line = None
    for i, line in enumerate(lines):
        if line == start_string:
            start_line = i
        elif line == end_string:
            end_line = i
            break

    with open(output_filepath, "w") as file:
        for i, line in enumerate(lines):
            if i > start_line and i < end_line:
                if line.startswith(f"{dimension} "):
                    continue
                else:
                    file.write(line)
            else:
                file.write(line)


def devsim_device_from_mesh(
    regions: set,
    interfaces: set,
    contact_interfaces: set,
    materials_dict: dict,
    interface_delimiter: str = "___",
    contact_delimiter: str = "@",
    mesh_name: str = "device_mesh",
    mesh_filepath: Path = "temp.msh2",
    device_name: str = "temp_device",
    reset: bool = False,
    regions_priority=None,  # for mesh fixing
    dimension: int = 2,
):
    # Initialize
    if reset:
        ds.reset_devsim()
    mesh_filename = str(mesh_filepath)
    nointerfaces_mesh_filename = str(Path(mesh_filepath).with_suffix(".fixed0.msh2"))
    fixed_mesh_filename = str(Path(mesh_filepath).with_suffix(".fixed1.msh2"))

    # Fix the mesh using scripts of https://github.com/devsim/devsim_misc/tree/main/gmsh
    # This currently adds redundancy with meshwell
    # First delete the existing interface physicals from the mesh file
    print("Fixing mesh")
    print("---------------------------------------------------------------")
    delete_interface_physicals(
        input_filepath=mesh_filename,
        output_filepath=nointerfaces_mesh_filename,
        dimension=dimension - 1,
    )
    get_fix_interfaces(
        nointerfaces_mesh_filename,
        fixed_mesh_filename,
        name_priority=tuple(regions_priority),
        interface_delimiter=interface_delimiter,
    )

    print("---------------------------------------------------------------")
    print("Creating mesh")
    print("---------------------------------------------------------------")
    ds.create_gmsh_mesh(mesh=mesh_name, file=fixed_mesh_filename)

    # Define the DEVSIM geometry
    print("---------------------------------------------------------------")
    print("Assigning regions")
    print("---------------------------------------------------------------")
    for region in regions:
        ds.add_gmsh_region(
            mesh=mesh_name,
            gmsh_name=region,
            region=region,
            material=materials_dict[region],
        )

    print("---------------------------------------------------------------")
    print("Assigning contacts")
    print("---------------------------------------------------------------")
    for contact_interface in contact_interfaces:
        contact_region, contacted_region, contact_port_name = parse_contact_interface(
            contact_interface=contact_interface,
            interface_delimiter=interface_delimiter,
            contact_delimiter=contact_delimiter,
        )
        if contacted_region == "None":
            continue
        ds.add_gmsh_contact(
            mesh=mesh_name,
            gmsh_name=contact_interface,
            region=contacted_region,
            material=materials_dict[contacted_region],
            name=contact_interface,
        )

    print("---------------------------------------------------------------")
    print("Assigning interfaces")
    print("---------------------------------------------------------------")
    for interface in interfaces:
        region0, region1 = interface.split(interface_delimiter)
        if region0 == "None" or region1 == "None":
            continue
        ds.add_gmsh_interface(
            gmsh_name=interface,
            mesh=mesh_name,
            name=interface,
            region0=region0,
            region1=region1,
        )

    print("---------------------------------------------------------------")
    print("Creating device")
    print("---------------------------------------------------------------")
    ds.finalize_mesh(mesh=mesh_name)
    ds.create_device(mesh=mesh_name, device=device_name)


if __name__ == "__main__":
    delete_interface_physicals(
        "../../docs/detectors/temp.msh2", "start_string", "end_string", "1"
    )
