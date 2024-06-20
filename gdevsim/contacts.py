def parse_contact_interface(contact_interface: str,
                            interface_delimiter: str = "___",
                            contact_delimiter: str = "@",
                            ):
    """
    Parses the contact interface string to extract the contact region, contacted region, and contact port name.

    Args:
        contact_interface (str): The string representing the contact interface.
        interface_delimiter (str): The delimiter used to separate the regions in the contact interface.
        contact_delimiter (str): The delimiter used to separate the contact region from the contact port name.

    Returns:
        tuple: A tuple containing the contact region, contacted region, and contact port name.
    """
    region0, region1 = contact_interface.split(interface_delimiter)
    contact_region, contacted_region = (
        (region0, region1) if contact_delimiter in region0 else (region1, region0)
    )
    contact_region, contact_port_name = contact_region.split(contact_delimiter)
    return f"{contact_region}{contact_delimiter}{contact_port_name}", contacted_region, contact_port_name
