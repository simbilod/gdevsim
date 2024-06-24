# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# # Insulators
#
# Insulators are defined as materials whose bandgaps >> relevant energy scales of the simulation. Hence, they only support bound charge, leaving only an electric field:
#
# * **Parameters**:
#     * Relative permittivity $\epsilon_r$ (unitless)
# * **Node Solutions**:
#     * Electrostatic potential $\varphi(\bm{x})$ (V)
# * **Equations**:
#     * Coulomb' law $\nabla^2 \varphi = 0$
#
# <div class="alert alert-success">
# Notes:
#
# * No free charge is assumed present
# </div>

# +
from gdevsim.materials.materials import get_all_materials

materials = get_all_materials()

for material_name, material in materials.items():
    if material["type"] == "insulator":
        if "eps_r" in material["general"]:
            print("=============================")
            print(material_name)
            print("=============================")
            permittivity = material["general"]["eps_r"]
            print(f"{'eps_r':<20} {permittivity}")
