from gdsfactory.technology import LayerLevel
from gdsfactory.typings import LayerStack

from gdevsim.utils.parse_layerstack import get_all_metal_layernames


def create_test_layerstack() -> LayerStack:
    return LayerStack(
        layers={
            "metal1": LayerLevel(material="copper"),
            "metal2": LayerLevel(material="aluminum"),
            "dielectric1": LayerLevel(material="silicon_dioxide"),
            "dielectric2": LayerLevel(material="silicon_nitride"),
        }
    )


def test_get_all_metal_layernames():
    layerstack = create_test_layerstack()
    materials_dict = {"copper": {"type": "metal"}, "aluminum": {"type": "metal"}, "silicon_dioxide": {"type": "dielectric"}, "silicon_nitride": {"type": "dielectric"}}
    assert get_all_metal_layernames(layerstack, materials_dict) == ["metal1", "metal2"]
