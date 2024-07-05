from gdsfactory.typings import Dict, LayerStack


def get_all_metal_layernames(layerstack: LayerStack,
                         materials_dict: Dict,
                         ignore: list[str],
                         ) -> list[str]:
    metal_layernames = []
    for layername, layer in layerstack.layers.items():
        if layername in ignore:
            continue
        else:
            if layer.material in materials_dict:
                if materials_dict[layer.material]["type"] == "metal":
                    metal_layernames.append(layername)
    return metal_layernames
