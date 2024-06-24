import shapely
from gdsfactory.typings import Component, Dict, LayerStack, Tuple
from gplugins.gmsh.uz_xsection_mesh import get_u_bounds_polygons

from gdevsim.doping.profile_impulses import function_mapping


def deserialize_profile(
    profile_dict,
    function_mapping: Dict = function_mapping,
):
    """Deserialize a doping impulse stored as a string in a LayerStack doping entry.

    Arguments:
        profile_dict: dict entry of LayerStack info containing:
            function_name: name of the function to use in function_mapping
            other keys: other keyword arguments of the function to use
        function_mapping: mapping between string and function
    """
    function_name = function_mapping[profile_dict["function"]]
    function_parameters = profile_dict["parameters"]
    return lambda x, y, ymax: function_name(x, y, ymax, **function_parameters)


def parse_layer_stack_doping_uz(
    layer_stack_dopings: LayerStack,
    layer_stack_geometry: LayerStack,
    component: Component,
    xsection_bounds: Tuple,
) -> Dict:
    """
    This function parses the layer stack doping information from the given layer stack dopings and component.

    TODO:
        * Use a dataclass for doping_data instead of a dict
        * Unit test

    Parameters:
        layer_stack_dopings (LayerStack): The layer stack dopings to parse.
        layer_stack_geometry (LayerStack): The layer stack containing the possible doping targets
        component (Component): The component to parse the layer stack dopings from.
        xsection_bounds (Tuple): used for uz cross-section

    Returns:
        dict: A dictionary containing the parsed layer stack doping information.
            The returned dictionary has the following entries:
            - "layer": The GDS layer of the doping.
            - "ion_types": The types of ions in the doping.
            - "into": The regions (layer stack keys) into which the doping is inserted.
            - "impulse_profiles": The impulse profiles of the doping.
            - "peak_concentrations": The peak concentrations of the doping.
            - "x_bounds": The x bounds of the doping.
    """
    # Parse layer stack doping information
    doping_data = {}

    # Data from layerstack
    for layer_name, layer in layer_stack_dopings.layers.items():
        doping_data[layer_name] = {
            "layer": layer.layer,
            "ion_types": layer.info["ion_types"],
            "into": layer.into,
            "impulse_profiles": layer.info["impulse_profiles"],
            "peak_concentrations": layer.info["peak_concentrations"],
        }

    # Data from polygons
    for doping_name, doping_dict in doping_data.items():
        doping_polygons = shapely.MultiPolygon(
            component.get_polygons(by_spec=doping_dict["layer"], as_shapely=True)
        )
        # Mask bounds is just mask
        mask_bounds = get_u_bounds_polygons(
            polygons=doping_polygons, xsection_bounds=xsection_bounds
        )
        # Implantation domain bounds is doping mask + physical implanted into
        domain_bounds = {}
        for into_region in doping_dict["into"]:
            if into_region in layer_stack_geometry.layers:
                into_layer = layer_stack_geometry.layers[into_region].layer
                structure_polygons = shapely.MultiPolygon(
                    component.get_polygons(by_spec=into_layer, as_shapely=True)
                )
                union = doping_polygons.union(structure_polygons)
                bounds = get_u_bounds_polygons(
                    polygons=union, xsection_bounds=xsection_bounds
                )
                domain_bounds[into_region] = bounds

        doping_data[doping_name]["domain_bounds"] = domain_bounds
        doping_data[doping_name]["mask_bounds"] = mask_bounds

    return doping_data




def parse_layer_stack_doping_xy(
    layer_stack_dopings: LayerStack,
    component: Component,
) -> Dict:
    """
    This function parses the layer stack doping information from the given layer stack dopings and component.

    TODO:
        * Use a dataclass for doping_data instead of a dict
        * Unit test

    Parameters:
        layer_stack_dopings (LayerStack): The layer stack dopings to parse.
        layer_stack_geometry (LayerStack): The layer stack containing the possible doping targets
        component (Component): The component to parse the layer stack dopings from.
        z (float): used for the plane

    Returns:
        dict: A dictionary containing the parsed layer stack doping information.
            The returned dictionary has the following entries:
            - "layer": The GDS layer of the doping.
            - "ion_types": The types of ions in the doping.
            - "into": The regions (layer stack keys) into which the doping is inserted.
            - "concentration": The concentrations of the doping.
            - "polygon_bounds": The xy polygon bounds of the doping.
    """
    # Parse layer stack doping information
    doping_data = {}

    # Data from layerstack
    for layer_name, layer in layer_stack_dopings.layers.items():
        doping_data[layer_name] = {
            "layer": layer.layer,
            "ion_types": layer.info["ion_types"],
            "into": layer.into,
            "concentrations": layer.info["peak_concentrations"],
        }

    # Data from polygons
    for doping_name, doping_dict in doping_data.items():
        doping_polygons = shapely.MultiPolygon(
            component.get_polygons(by_spec=doping_dict["layer"], as_shapely=True)
        )
        doping_data[doping_name]["mask_bounds"] = doping_polygons

    return doping_data
