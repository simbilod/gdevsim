"""Adapted from DEVSIM LLC https://github.com/devsim/devsim/blob/main/python_packages/simple_physics.py

Improvements:
    - more Pythonic names
"""
import devsim as ds


def set_interface_parameters(
    device, region, parameters=None, physics=None, interface_delimiter="___"
):
    """Interface-dependent parameters are stored in parameters["interfaces"]

    Convention: append at_interface_name to parameter name
    """
    # All physics keys (interface physics can appear everywhere)
    physics_keys = set()
    for category, subdict in physics.items():
        physics_keys.add(category)
        for effect in subdict.values():
            for sub_effect in effect:
                physics_keys.add(sub_effect)

    # For each interface
    current_interfaces = [
        interface
        for interface in ds.get_interface_list(device=device)
        if region in interface
    ]

    # Tag interfaces by materials
    for interface in current_interfaces:
        region0, region1 = interface.split(interface_delimiter)
        material0 = ds.get_material(device=device, region=region0)
        material1 = ds.get_material(device=device, region=region1)
        bulk_material = ds.get_material(device=device, region=region)
        other_material = material0 if material0 != bulk_material else material1
        if not material0 == material1:
            at_interface_name = f"region_{region}_at_interface_{interface}"

            # If the requisite physics is present
            if "interfaces" in parameters:
                for physics, physics_parameters in parameters["interfaces"].items():
                    if physics in physics_keys:
                        # If the other material is in the database
                        for other_material_name in physics_parameters:
                            if other_material_name == other_material:
                                for (
                                    parameter_name,
                                    parameter_value,
                                ) in physics_parameters[other_material_name].items():
                                    ds.set_parameter(
                                        device=device,
                                        region=region,
                                        name=f"{parameter_name}_{at_interface_name}",
                                        value=parameter_value,
                                    )


def set_parameters(device, region, parameters, physics):
    # Set global parameters
    for parameter_name, parameter_value in parameters.items():
        if parameter_name not in list(physics) + ["general", "type"]:
            ds.set_parameter(
                device=device,
                region=region,
                name=parameter_name,
                value=parameter_value,
            )
    # Set general parameters
    for parameter_name, parameter_value in parameters["general"].items():
        ds.set_parameter(
            device=device,
            region=region,
            name=parameter_name,
            value=parameter_value,
        )
    # Set physics-specific parameters
    for effect in physics:
        for effect_name, effect_parameters in parameters[effect].items():
            if isinstance(effect_parameters, dict):
                for parameter_name, parameter_value in effect_parameters.items():
                    ds.set_parameter(
                    device=device,
                    region=region,
                    name=parameter_name,
                    value=parameter_value,
                )
            else:
                ds.set_parameter(
                    device=device,
                    region=region,
                    name=effect_name,
                    value=effect_parameters,
                )