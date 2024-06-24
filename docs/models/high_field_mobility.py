# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: devsim
#     language: python
#     name: python3
# ---

# # Mobility models
#
# Mobility relates drift velocity to electric field:
#
# $$ v_d = \mu E $$
#
# This is only valid for low enough fields, since infinite drift velocities are not observed.
#
# In this notebook, we explain different high field mobility mobels implemented in gdevsim, and show that their value after being instanciated in simulations agrees with literature. They are based on a modification of the low-mobility $\mu_{low}$ according to a driving force $F$:
#
# $$ \mu = f(\mu_{low}, F) $$
#
# Below, $T_{ref}$ is always taken to be $300$ K.

import gdsfactory as gf
import numpy as np
import ray
from gdsfactory.technology import LayerLevel, LayerStack
from gdsfactory.typings import Tuple

from gdevsim.materials.materials import get_all_materials, get_global_parameters
from gdevsim.simulation import DevsimComponent

materials = get_all_materials()


@ray.remote
def mobility_from_simulation(material: str,
                        mobility: Tuple[str],
                        T: float,
                        channel_width: float = 1,
                        ):
    """Test object to check how mobility is implemented in simulations.

    Arguments:
        material: material name (str)
        mobility: mobility models to use
        T: temperature (K)
    """

    # Create test stack
    layer_stack_for_testing = LayerStack(
        layers=dict(
            channel=LayerLevel(
                layer=(1,0),
                thickness=1,
                zmin=0,
                material=material,
                mesh_order=2,
            ),
            contact=LayerLevel(
                layer=(2,0),
                thickness=1,
                zmin=0,
                material="al",
                mesh_order=1,
            ),
        )
    )

    # Create test component
    @gf.cell
    def component_for_testing(channel_width: float = channel_width):
        c = gf.Component()
        channel = c << gf.components.rectangle(size=(channel_width,1), layer=(1,0))
        left_contact = c << gf.components.rectangle(size=(1,1), layer=(2,0))
        right_contact = c << gf.components.rectangle(size=(1,1), layer=(2,0))
        left_contact.connect("e3", destination=channel.ports["e1"])
        right_contact.connect("e1", destination=channel.ports["e3"])
        c.add_ports("e_left", port=left_contact.ports["e1"])
        c.add_ports("e_right", port=right_contact.ports["e3"])
        return c

    # Set temperature
    global_parameters = get_global_parameters()
    global_parameters["T"] = T

    physics_parameters = {
            material: {
                "mobility": mobility,
            },
        }

    # Reinitialize simulation
    simulation = DevsimComponent(
        component=component_for_testing,
        layer_stack=layer_stack_for_testing,
        mesh_type="xy",
        z=0.5,
        global_parameters=global_parameters,
        physics_parameters=physics_parameters
    )
    simulation.reset()

    # Read instanciated mobilities
    simulation.initialize(
        default_characteristic_length=1.0,
        reset_save_directory=True,
        threads_available=1,
    )

    n_mobility = simulation.get_node_field_values(field="mu_n_node_lf", regions=["channel"])[0]
    p_mobility = simulation.get_node_field_values(field="mu_p_node_lf", regions=["channel"])[0]

    # Delete files to avoid clutter
    simulation.reset()

    # Return data
    return T, n_mobility, p_mobility


# ## `highfield_canali`
#
# The Canali model modifies the Caughey-Thomas formula with temperature-dependent parameters. It turns a low-field mobility $\mu_{low}$ from Notebook 3 into a function of driving force as:
#
# $$ \mu(F) = \mu_{low} \left( 1 + \left( \frac{\mu_{low} F}{v_{sat}} \right)^{\beta} \right)^{-\frac{1}{\beta}} $$
#
# Each parameter has temperature dependence:
#
# $$ v_{sat} = v_{sat,0} \left( \frac{T}{T_{ref}} \right)^{v_{sat,exp}} $$
# $$ \beta = \beta_0 \left( \frac{T}{T_{ref}} \right)^{\beta_{exp}} $$
#
# A separate expression and set of parameters is used for each carrier type (electrons $n$ and holes $p$).

for material_name, material in materials.items():
    if material["type"] == "semiconductor" and "mobility" in material:
        if "highfield_canali" in material["mobility"]:
            print("=============================")
            print(material_name)
            print("=============================")
            doping_arora = material["mobility"]["highfield_canali"]
            for key, value in doping_arora.items():
                print(f"{key:<20} {value}")

# +
temperatures = [300, 370, 430]
voltages = np.linspace(0,10,11)

materials = ["silicon", "germanium"]
mobility_calculations = []

for temperature in temperatures:
    for _voltage in voltages:
        for material in materials:
            mobility_calculations.append(mobility_from_simulation.remote(material=material,
                                mobility=("doping_arora", "highfield_canali"),
                                T=temperature,
                                )
                            )
# -

# <!-- # ## Bibliography
# #
# # ```{bibliography}
# # :style: unsrt
# # :filter: docname in docnames
# # ``` -->
