import devsim as ds
import numpy as np
import shapely
from devsim import set_parameter

from gdevsim.contacts import parse_contact_interface
from gdevsim.models import assign_parameters, models
from gdevsim.models.interpolation import add_structured_data_to_mesh


def assign_interface_models(
    device_name, interface, materials_parameters, materials_dict
):
    region1, region2 = interface.split("___")

    if region1 != "None" and region2 != "None":

        material1_type = materials_parameters[materials_dict[region1]]["type"]
        material2_type = materials_parameters[materials_dict[region2]]["type"]

        print(f"Assigning models to interface {interface} of type {material1_type}-{material2_type}")

        if (material1_type == "insulator" and material2_type == "semiconductor") or (
            material1_type == "semiconductor" and material2_type == "insulator"
        ):
            models.create_potentialOnly_interface(
                device=device_name, interface=interface
            )
        elif material1_type == "insulator" and material2_type == "insulator":
            models.create_potentialOnly_interface(
                device=device_name, interface=interface
            )
        elif material1_type == "semiconductor" and material2_type == "semiconductor":
            models.create_semiconductor_semiconductor_interface(
                device=device_name, interface=interface
            )



def assign_region_models(
    region,
    materials_parameters,
    materials_dict,
    global_parameters,
    physics,
    device_name,
    interface_delimiter="___",
):

    print(f"Assigning parameters to bulk nodes and interface nodes of region {region} of material {materials_dict[region]}")
    assign_parameters.set_interface_parameters(
        device=device_name,
        region=region,
        parameters=materials_parameters[materials_dict[region]],
        physics=physics,
        interface_delimiter=interface_delimiter,
    )
    opts = {}
    # Assign parameters
    assign_parameters.set_parameters(
        device=device_name,
        region=region,
        parameters={
            **global_parameters,
            **materials_parameters[materials_dict[region]],
        },
        physics=physics[materials_dict[region]],
    )

    print(f"Assigning models to region {region} of type {materials_parameters[materials_dict[region]]['type']}")

    if materials_parameters[materials_dict[region]]["type"] == "insulator":
        models.create_insulator_potential(device=device_name, region=region)

    elif materials_parameters[materials_dict[region]]["type"] == "semiconductor":
        models.create_semiconductor_potential(
            device=device_name, region=region, physics=physics[materials_dict[region]]
        )
        opts[region] = models.create_semiconductor_drift_diffusion(
            device=device_name,
            region=region,
            physics=physics[materials_dict[region]],
            interface_delimiter=interface_delimiter,
        )
    # else:  # TODO: add metal

    return opts


def assign_contact_models(
    contact_interface,
    device_name,
    materials_parameters,
    materials_dict,
    interface_delimiter,
    contact_delimiter,
    opts,
):
    contact_region, contacted_region, contact_port_name = parse_contact_interface(
        contact_interface=contact_interface,
        interface_delimiter=interface_delimiter,
        contact_delimiter=contact_delimiter,
    )
    set_parameter(
        device=device_name,
        name=models.get_contact_bias_name(contact_interface),
        value=0.0,
    )

    print(f"Assigning models to contact port {contact_port_name} contacting from {contact_region} to {contacted_region}")

    if contacted_region != "None":
        if (
            materials_parameters[materials_dict[contacted_region]]["type"]
            == "semiconductor"
        ):
            models.create_semiconductor_potentialOnly_contact(
                device=device_name, region=contacted_region, contact=contact_interface
            )
            models.create_semiconductor_drift_diffusion_contact(
                device=device_name,
                region=contacted_region,
                contact=contact_interface,
                **opts[contacted_region],
            )
        elif (
            materials_parameters[materials_dict[contacted_region]]["type"]
            == "insulator"
        ):
            models.create_insulator_potentialOnly_contact(
                device=device_name, region=contacted_region, contact=contact_interface
            )


def assign_doping_profiles_uz(
    regions,
    materials_dict,
    materials_parameters,
    total_profiles,
    layer_stack_geometry,
    device_name,
    global_scaling,
):
    for region in regions:

        # Extract background doping
        background_ion = layer_stack_geometry.layers[
            region
        ].background_doping_ion
        background_concentration = layer_stack_geometry.layers[
            region
        ].background_doping_concentration

        material_name = materials_dict[region]
        if materials_parameters[material_name]["type"] == "semiconductor":
            if region in total_profiles:
                x_samplings = total_profiles[region]["x_samplings"]
                y_samplings = total_profiles[region]["y_samplings"]

                # Also add background
                if background_ion and background_concentration:
                    if background_ion == "donor":
                        total_profiles[region]["donor"] += background_concentration
                    elif background_ion == "acceptor":
                        total_profiles[region]["acceptor"] += background_concentration
                    else:
                        raise ValueError(
                            f"background_doping_ion for layer {region} = {background_ion} not one of donor or acceptor!"
                        )

                # Add acceptors
                add_structured_data_to_mesh(
                    device=device_name,
                    region=region,
                    name="Acceptors",
                    x_array=x_samplings * global_scaling,  # match units
                    y_array=y_samplings * global_scaling,  # match units
                    z_array=None,
                    val_array=total_profiles[region]["acceptor"],
                )

                # Add donors
                add_structured_data_to_mesh(
                    device=device_name,
                    region=region,
                    name="Donors",
                    x_array=x_samplings * global_scaling,  # match units
                    y_array=y_samplings * global_scaling,  # match units
                    z_array=None,
                    val_array=total_profiles[region]["donor"],
                )

                # Add net doping
                ds.node_model(
                    device=device_name,
                    region=region,
                    name="NetDoping",
                    equation="Donors-Acceptors",
                )

            # If just background, add constant
            elif background_ion and background_concentration:
                if background_ion == "donor":
                    donor_values = background_concentration
                    acceptors_values = 0
                elif background_ion == "acceptor":
                    donor_values = 0
                    acceptors_values = background_concentration
                else:
                    raise ValueError(
                        f"background_doping_ion for layer {region} = {background_ion} not one of donor or acceptor!"
                    )
                nodes = ds.get_node_model_values(device=device_name, region=region, name="x")
                ds.node_solution(device=device_name, region=region, name="Donors")
                ds.set_node_values(device=device_name, region=region, name="Donors", values=[donor_values] * len(nodes))
                ds.node_solution(device=device_name, region=region, name="Acceptors")
                ds.set_node_values(device=device_name, region=region, name="Acceptors", values=[acceptors_values] * len(nodes))

                # Add net doping
                ds.node_model(
                    device=device_name,
                    region=region,
                    name="NetDoping",
                    equation="Donors-Acceptors",
                )

            # Else no doping
            else:

                nodes = ds.get_node_model_values(device=device_name, region=region, name="x")
                ds.node_solution(device=device_name, region=region, name="Donors")
                ds.set_node_values(device=device_name, region=region, name="Donors", values=[0] * len(nodes))
                ds.node_solution(device=device_name, region=region, name="Acceptors")
                ds.set_node_values(device=device_name, region=region, name="Acceptors", values=[0] * len(nodes))

                # Add net doping
                ds.node_model(
                    device=device_name,
                    region=region,
                    name="NetDoping",
                    equation="Donors-Acceptors",
                )


def assign_doping_profiles_xy(
    regions,
    materials_dict,
    materials_parameters,
    doping_data,
    layer_stack_geometry,
    device_name,
    global_scaling,
):
    for region in regions:

        # Initialize models
        ds.node_solution(device=device_name, region=region, name="Acceptors")
        ds.node_solution(device=device_name, region=region, name="Donors")

        # Extract background doping
        background_ion = layer_stack_geometry.layers[
            region
        ].background_doping_ion
        background_concentration = layer_stack_geometry.layers[
            region
        ].background_doping_concentration

        material_name = materials_dict[region]
        if materials_parameters[material_name]["type"] == "semiconductor":

            # Accumulate concentrations across doping data
            xpos = np.array(ds.get_node_model_values(device=device_name, region=region, name="x")) / global_scaling
            ypos = np.array(ds.get_node_model_values(device=device_name, region=region, name="y")) / global_scaling
            points = shapely.points(coords=xpos, y=ypos)
            acceptors = np.zeros_like(xpos)
            donors = np.zeros_like(xpos)

            for _doping_name, doping_dict in doping_data.items():
                mask_polygon = doping_dict["mask_bounds"]
                if region in doping_dict["into"]:
                    for concentration, ion_type in zip(doping_dict["concentrations"], doping_dict["ion_types"]):
                        if ion_type == "donor":
                            donors += np.where(mask_polygon.contains(points), concentration, 0)
                        if ion_type == "acceptor":
                            acceptors += np.where(mask_polygon.contains(points), concentration, 0)

            # Also add background
            if background_ion and background_concentration:
                if background_ion == "donor":
                    donors += background_concentration
                elif background_ion == "acceptor":
                    acceptors += background_concentration
                else:
                    raise ValueError(
                        f"background_doping_ion for layer {region} = {background_ion} not one of donor or acceptor!"
                    )

            # Add acceptors
            ds.set_node_values(device=device_name, region=region, name="Acceptors", values=acceptors)
            # Add donors
            ds.set_node_values(device=device_name, region=region, name="Donors", values=donors)
            # Add net doping
            ds.node_model(
                device=device_name,
                region=region,
                name="NetDoping",
                equation="Donors-Acceptors",
            )
