"""Basic material information"""
import os

import yaml

from gdevsim.config import PATH


def get_global_parameters():
    return {
        "k": 1.3806503e-23,  # JK-1
        # "k": 8.617e-5,  # eVK-1
        "q": 1.6e-19,  # Coulombs
        "eps_0": 8.85e-14,  # Fcm-2
        "T": 300,  # K
    }


def get_material(material_name):
    with open(os.path.join(PATH.materials, f"{material_name}.yaml")) as file:
        material_properties = yaml.safe_load(file)
    return material_properties


def get_all_materials():
    """Returns all materials in the database. Used as a default argument in simulation initialization."""
    material_properties = {}
    for file in [file for file in os.listdir(PATH.materials) if file.endswith(".yaml")]:
        material_name = os.path.splitext(file)[0]
        material_properties[material_name] = get_material(material_name)

    return material_properties


def get_default_physics():
    """Returns a dict toggling material-dependent physics."""

    return {
        "silicon": {
            "bandstructure": ("bandgapnarrowing_slotboom",), 
            "mobility": ("doping_arora", "highfield_canali"),
            "generation_recombination": ("bulkSRH"),
        },
        "germanium": {
            "bandstructure": ("bandgapnarrowing_slotboom",), 
            "mobility": ("doping_arora", "highfield_canali"),
            "generation_recombination": ("bulkSRH", "optical_generation"),
            "interfaces": ("surfaceSRH",),
        },
        "aluminum": {},
        "silicon_dioxide": {},
        "copper": {},
        "silicon_nitride": {},
        "silicon_insulator": {},
    }


if __name__ == "__main__":
    print(get_all_materials())