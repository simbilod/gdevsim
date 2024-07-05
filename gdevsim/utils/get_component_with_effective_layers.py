import copy

import gdsfactory as gf
import gdstk
import shapely
from gdsfactory.typings import Component, Dict, Layers, LayerStack, List


def get_component_layer_stack(
    component: Component,
    layer_stack: LayerStack,
    additional_layers: Layers | None = None
) -> LayerStack:
    """Returns a new layer_stack only with layers that appear in the provided component.

    Arguments:
        component: to process.
        layer_stack: to process.
        add_layers: extra layers to manually include (e.g. WAFER)

    Returns:
        new_layer_stack: without layers that do not appear in component.
    """
    new_layer_stack = layer_stack.model_copy()

    layers_present = component.layers
    layernames_dict = new_layer_stack.get_layer_to_layername()
    layernames_present = [
        name
        for sublist in [layernames_dict[layer] for layer in layers_present]
        for name in sublist
    ]
    if additional_layers is not None:
        layernames_present += [
            name
            for sublist in [layernames_dict[layer] for layer in additional_layers]
                for name in sublist
            ]
    for key in list(new_layer_stack.layers.keys()):
        if key not in layernames_present + ["box", "clad"]:
            new_layer_stack.layers.pop(key)

    return new_layer_stack


def get_component_with_net_layers(
    component,
    layer_stack,
    port_names: list[str],
    delimiter: str = "#",
    new_layers_init: tuple[int, int] = (10010, 0),
    add_to_layerstack: bool = True,
    additional_layers: Layers | None = None
) -> Component:
    """Uses port's layer attribute to decide which polygons need to be renamed.
    New layers are named "layername{delimiter}portname".

    Automatically increments the new layer numbers from new_layers_init.

    Args:
        component: to process.
        layer_stack: to process.
        port_names: list of port_names to process into new layers.
        delimiter: the new layer created is called "layername{delimiter}portname".
        new_layers_init: initial layer number for the temporary new layers.
        add_to_layerstack: True by default, but can be set to False to disable parsing of the layerstack.
        additional_layernames: extra layernames to manually include (e.g. WAFER)

    Returns:
        net_component: component with port polygons replaces
        net_layer_stack: layer_stack with new entries for the port polygons
        port_map: mapping between port_name and net_layer_stack label representing the port polygons
    """
    # Initialize returned component and layerstack
    net_component = gf.get_component(component).copy()
    net_layer_stack = layer_stack.model_copy()
    port_map = {}

    # For each port to consider, convert relevant polygons
    for i, portname in enumerate(port_names):
        port = component.ports[portname]

        # Get original port layer polygons, and modify a new component without that layer
        polygons = net_component.extract(layers=[port.layer]).get_polygons()
        net_component = net_component.remove_layers(layers=[port.layer])

        for polygon in polygons:
            # If polygon belongs to port, create a unique new layer, and add the polygon to it
            if gdstk.inside(
                [port.center],
                gdstk.offset(gdstk.Polygon(polygon), gf.get_active_pdk().grid_size),
            )[0]:
                try:
                    port_layernames = layer_stack.get_layer_to_layername()[port.layer]
                except KeyError as e:
                    raise KeyError(
                        "Make sure your `layer_stack` contains all layers with ports"
                    ) from e
                for j, old_layername in enumerate(port_layernames):
                    new_layer_number = (
                        new_layers_init[0] + i,
                        new_layers_init[1] + j,
                    )
                    if add_to_layerstack:
                        new_layer = copy.deepcopy(net_layer_stack.layers[old_layername])
                        new_layer.layer = (
                            new_layers_init[0] + i,
                            new_layers_init[1] + j,
                        )
                        net_layer_stack.layers[
                            f"{old_layername}{delimiter}{portname}"
                        ] = new_layer
                        port_map[portname] = f"{old_layername}{delimiter}{portname}"
                    net_component.add_polygon(polygon, layer=new_layer_number)
            # Otherwise put the polygon back on the same layer
            else:
                net_component.add_polygon(polygon, layer=port.layer)

    net_component.name = f"{component.name}_net_layers"
    return net_component, get_component_layer_stack(net_component, net_layer_stack, additional_layers=additional_layers), port_map



def get_component_with_propagated_net_layers(component: Component,
                                       layer_stack: LayerStack,
                                       propagate_layers: Dict[str, List[str]] | None = None,
                                       new_layers_init: tuple[int, int] = (20010, 0),
                                       additional_layers: Layers | None = None,
                                       delimiter: str = "#",
                                       ) -> tuple[Component, LayerStack, Dict[str, List[str]]]:
    """Returns a new component with layers that connect to the provided component."""
    # Initialize returned component and layerstack
    net_component = gf.get_component(component).copy()
    net_layer_stack = layer_stack.model_copy()
    layer_physical_map = {}

    i = 0
    for source_layer, destination_layers in propagate_layers.items():
        source_polygon = net_component.extract(layers=[layer_stack.layers[source_layer].layer]).get_polygons()
        source_polygon_shapely = shapely.geometry.Polygon(source_polygon[0])
        all_possible_destination_polygons = net_component.extract(layers=[layer_stack.layers[layer].layer for layer in destination_layers]).get_polygons()
        for possible_destination_polygon in all_possible_destination_polygons:
            possible_destination_polygon_shapely = shapely.geometry.Polygon(possible_destination_polygon)
            if source_polygon_shapely.intersects(possible_destination_polygon_shapely):
                # We found which set of unified shapes touch the contact; process component
                for layer in destination_layers:
                    # Create new logical layer (if not already present)
                    if f"{layer}{delimiter}{source_layer}" not in net_layer_stack.layers:
                        layer_number = layer_stack.layers[layer].layer
                        new_layer_number = (
                            new_layers_init[0] + i,
                            0,
                            )
                        new_layer = copy.deepcopy(net_layer_stack.layers[layer])
                        new_layer.layer = (
                            new_layers_init[0] + i,
                            0,
                        )
                        i += 1
                        net_layer_stack.layers[f"{layer}{delimiter}{source_layer}"] = new_layer
                        layer_physical_map[f"{layer}{delimiter}{source_layer}"] = source_layer

                    # Possibly put relevant polygons on that layer
                    polygons = net_component.extract(layers=[layer_number]).get_polygons()
                    net_component = net_component.remove_layers(layers=[layer_number])
                    for polygon in polygons:
                        polygon_shapely = shapely.geometry.Polygon(polygon)
                        if polygon_shapely.intersects(possible_destination_polygon_shapely):
                            net_component.add_polygon(polygon, layer=new_layer_number)
                        # Otherwise put the polygon back on the same layer
                        else:
                            net_component.add_polygon(polygon, layer=layer_number)

    return net_component, get_component_layer_stack(net_component, net_layer_stack, additional_layers=additional_layers), layer_physical_map

if __name__ == "__main__":
    c, ls = get_component_with_net_layers()
    c.show()
    print(ls.keys())
