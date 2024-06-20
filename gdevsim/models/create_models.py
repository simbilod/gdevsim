"""Adapted from J. Sanchez and DEVSIM LLC:
    - https://github.com/devsim/devsim/blob/main/python_packages/model_create.py
    - https://github.com/devsim/devsim_bjt_example/blob/main/simdir/physics/model_create.py

Improvements:
    - more Pythonic names
    - cleaner imports
    - f-strings
"""

import devsim as ds

debug = False


def create_solution(device: str, region: str, name: str) -> None:
    """
    Create solution variables for a specified device and region, and initialize their values on each edge.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        name (str): The name of the solution variable to create.
    """
    ds.node_solution(name=name, device=device, region=region)
    ds.edge_from_node_model(node_model=name, device=device, region=region)


def create_node_model(device: str, region: str, model: str, expression: str) -> None:
    """
    Creates a node model with the given expression for a specified device and region.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the node model to create.
        expression (str): The mathematical expression defining the node model.
    """
    result = ds.node_model(
        device=device, region=region, name=model, equation=expression
    )
    if debug:
        print(f'NODEMODEL {device} {region} {model} "{result}"')


def create_node_model_derivative(
    device: str, region: str, model: str, expression: str, *vars: str
) -> None:
    """
    Create a node model derivative.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the node model to create.
        expression (str): The mathematical expression defining the node model.
        vars (str): Variables with respect to which the derivative will be taken.
    """
    for v in vars:
        create_node_model(device, region, f"{model}:{v}", f"diff({expression},{v})")


def create_contact_node_model(
    device: str, contact: str, model: str, expression: str
) -> None:
    """
    Creates a contact node model.

    Args:
        device (str): The name of the device.
        contact (str): The name of the contact.
        model (str): The name of the node model to create.
        expression (str): The mathematical expression defining the node model.
    """
    result = ds.contact_node_model(
        device=device, contact=contact, name=model, equation=expression
    )
    if debug:
        print(f'CONTACTNODEMODEL {device} {contact} {model} "{result}"')


def create_contact_node_model_derivative(
    device: str, contact: str, model: str, expression: str, variable: str
) -> None:
    """
    Creates a contact node model derivative.

    Args:
        device (str): The name of the device.
        contact (str): The name of the contact.
        model (str): The name of the node model to create.
        expression (str): The mathematical expression defining the node model.
        variable (str): The variable with respect to which the derivative will be taken.
    """
    create_contact_node_model(
        device, contact, f"{model}:{variable}", f"diff({expression}, {variable})"
    )


def create_edge_model(device: str, region: str, model: str, expression: str) -> None:
    """
    Creates an edge model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the edge model to create.
        expression (str): The mathematical expression defining the edge model.
    """
    result = ds.edge_model(
        device=device, region=region, name=model, equation=expression
    )
    if debug:
        print(f'EDGEMODEL {device} {region} {model} "{result}"')


def create_edge_model_derivatives(
    device: str, region: str, model: str, expression: str, variable: str
) -> None:
    """
    Creates edge model derivatives.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the edge model to create.
        expression (str): The mathematical expression defining the edge model.
        variable (str): The variable with respect to which the derivative will be taken.
    """
    create_edge_model(
        device, region, f"{model}:{variable}@n0", f"diff({expression}, {variable}@n0)"
    )
    create_edge_model(
        device, region, f"{model}:{variable}@n1", f"diff({expression}, {variable}@n1)"
    )


def create_contact_edge_model(
    device: str, contact: str, model: str, expression: str
) -> None:
    """
    Creates a contact edge model.

    Args:
        device (str): The name of the device.
        contact (str): The name of the contact.
        model (str): The name of the edge model to create.
        expression (str): The mathematical expression defining the edge model.
    """
    result = ds.contact_edge_model(
        device=device, contact=contact, name=model, equation=expression
    )
    if debug:
        print(f'CONTACTEDGEMODEL {device} {contact} {model} "{result}"')


def create_contact_edge_model_derivative(
    device: str, contact: str, model: str, expression: str, variable: str
) -> None:
    """
    Creates contact edge model derivatives with respect to variable on node.

    Args:
        device (str): The name of the device.
        contact (str): The name of the contact.
        model (str): The name of the edge model to create.
        expression (str): The mathematical expression defining the edge model.
        variable (str): The variable with respect to which the derivative will be taken.
    """
    create_contact_edge_model(
        device, contact, f"{model}:{variable}", f"diff({expression}, {variable})"
    )


def create_interface_model(
    device: str, interface: str, model: str, expression: str
) -> None:
    """
    Creates a interface node model.

    Args:
        device (str): The name of the device.
        interface (str): The name of the interface.
        model (str): The name of the node model to create.
        expression (str): The mathematical expression defining the node model.
    """
    result = ds.interface_model(
        device=device, interface=interface, name=model, equation=expression
    )
    if debug:
        print(f'INTERFACEMODEL {device} {interface} {model} "{result}"')


def create_continuous_interface_model(
    device: str, interface: str, variable: str
) -> str:
    """
    Creates a continuous interface model.

    Args:
        device (str): The name of the device.
        interface (str): The name of the interface.
        variable (str): The variable to be used in the model.

    Returns:
        str: The name of the created model.
    """
    mname = f"continuous{variable}"
    meq = f"{variable}@r0 - {variable}@r1"
    mname0 = f"{mname}:{variable}@r0"
    mname1 = f"{mname}:{variable}@r1"
    create_interface_model(device, interface, mname, meq)
    create_interface_model(device, interface, mname0, "1")
    create_interface_model(device, interface, mname1, "-1")
    return mname


def in_edge_model_list(device: str, region: str, model: str) -> bool:
    """
    Checks to see if this edge model is available on device and region.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the edge model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    return model in ds.get_edge_model_list(device=device, region=region)


def in_node_model_list(device: str, region: str, model: str) -> bool:
    """
    Checks to see if this node model is available on device and region.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the node model to check.

    Returns:
        bool: True if the model exists, False otherwise.
    """
    return model in ds.get_node_model_list(device=device, region=region)


def ensure_edge_from_node_model_exists(
    device: str, region: str, nodemodel: str
) -> None:
    """
    Checks if the edge models exists.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        nodemodel (str): The name of the node model to check.

    Raises:
        ValueError: If the node model does not exist.
    """
    if not in_node_model_list(device, region, nodemodel):
        raise ValueError(f"{nodemodel} must exist")

    ds.get_edge_model_list(device=device, region=region)
    emtest = f"{nodemodel}@n0" and f"{nodemodel}@n1"
    if not emtest:
        if debug:
            print(f"INFO: Creating {nodemodel}@n0 and {nodemodel}@n1")
        ds.edge_from_node_model(device=device, region=region, node_model=nodemodel)


def create_element_model2d(
    device: str, region: str, model: str, expression: str
) -> None:
    """
    Creates a 2D element model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the element model to create.
        expression (str): The mathematical expression defining the element model.
    """
    result = ds.element_model(
        device=device, region=region, name=model, equation=expression
    )
    if debug:
        print(f'ELEMENTMODEL {device} {region} {model} "{result}"')


def create_element_model_derivative2d(
    device: str, region: str, model_name: str, expression: str, *args: str
) -> None:
    """
    Creates a 2D element model derivative.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model_name (str): The name of the element model to create.
        expression (str): The mathematical expression defining the element model.
        args (str): Variables with respect to which the derivative will be taken.

    Raises:
        ValueError: If no variable names are provided.
    """
    if len(args) == 0:
        raise ValueError("Must specify a list of variable names")
    for i in args:
        for j in ("@en0", "@en1", "@en2"):
            create_element_model2d(
                device, region, f"{model_name}:{i}{j}", f"diff({expression}, {i}{j})"
            )


def create_geometric_mean(device: str, region: str, nmodel: str, emodel: str) -> None:
    """
    Creates a geometric mean model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        nmodel (str): The name of the node model.
        emodel (str): The name of the edge model.
    """
    ds.edge_average_model(
        device=device,
        region=region,
        edge_model=emodel,
        node_model=nmodel,
        average_type="geometric",
    )


def create_geometric_mean_derivative(
    device: str, region: str, nmodel: str, emodel: str, *args: str
) -> None:
    """
    Creates a geometric mean derivative model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        nmodel (str): The name of the node model.
        emodel (str): The name of the edge model.
        args (str): Variables with respect to which the derivative will be taken.

    Raises:
        ValueError: If no variable names are provided.
    """
    if len(args) == 0:
        raise ValueError("Must specify a list of variable names")
    for i in args:
        ds.edge_average_model(
            device=device,
            region=region,
            edge_model=emodel,
            node_model=nmodel,
            derivative=i,
            average_type="geometric",
        )


def create_arithmetic_mean(device: str, region: str, nmodel: str, emodel: str) -> None:
    """
    Creates an arithmetic mean model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        nmodel (str): The name of the node model.
        emodel (str): The name of the edge model.
    """
    ds.edge_average_model(
        device=device,
        region=region,
        edge_model=emodel,
        node_model=nmodel,
        average_type="arithmetic",
    )


def create_arithmetic_mean_derivative(
    device: str, region: str, nmodel: str, emodel: str, *args: str
) -> None:
    """
    Creates an arithmetic mean derivative model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        nmodel (str): The name of the node model.
        emodel (str): The name of the edge model.
        args (str): Variables with respect to which the derivative will be taken.

    Raises:
        ValueError: If no variable names are provided.
    """
    if len(args) == 0:
        raise ValueError("Must specify a list of variable names")
    for i in args:
        ds.edge_average_model(
            device=device,
            region=region,
            edge_model=emodel,
            node_model=nmodel,
            derivative=i,
            average_type="arithmetic",
        )


def interface_normal_model(
    device: str, region: str, model: str, expression: str
) -> None:
    """
    Creates an edge model.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the edge model to create.
        expression (str): The mathematical expression defining the edge model.
    """
    result = ds.edge_model(
        device=device, region=region, name=model, equation=expression
    )
    if debug:
        print(f'EDGEMODEL {device} {region} {model} "{result}"')


def create_interface_normal_model_derivatives(
    device: str, region: str, model: str, expression: str, variable: str
) -> None:
    """
    Creates edge model derivatives.

    Args:
        device (str): The name of the device.
        region (str): The name of the region within the device.
        model (str): The name of the edge model to create.
        expression (str): The mathematical expression defining the edge model.
        variable (str): The variable with respect to which the derivative will be taken.
    """
    create_edge_model(
        device, region, f"{model}:{variable}@n0", f"diff({expression}, {variable}@n0)"
    )
    create_edge_model(
        device, region, f"{model}:{variable}@n1", f"diff({expression}, {variable}@n1)"
    )
