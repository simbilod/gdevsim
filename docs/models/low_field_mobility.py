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
# Mobility captures the ability of a charge carrier to move in a material under the influence of an electric field. 
#
# In this notebook, we explain different low field mobility mobels implemented in gdevsim, and show that their value after being instanciated in simulations agrees with literature.
#
# Below, $T_{ref}$ is always taken to be $300$ K.

# +
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import ray
import gdsfactory as gf
from gdsfactory.typings import Tuple
from gdsfactory.technology import LayerLevel, LayerStack
from gdevsim.simulation import DevsimComponent
from gdevsim.config import Path
from gdevsim.materials.materials import get_all_materials
from gdevsim.materials.materials import get_global_parameters, get_default_physics
materials = get_all_materials()

ray.init(log_to_driver=False)


# -

@ray.remote
def mobility_from_simulation(material: str, 
                        mobility: Tuple[str],
                        T: float, 
                        doping_conc: float, 
                        doping_type: str,
                        ):
    """Test object to check how mobility is implemented in simulations.

    Arguments:
        material: material name (str)
        mobility: mobility models to use
        T: temperature (K)
        doping_conc: doping concentration (cm^-3)
        doping_type: type of doping
    """

    # Create test stack
    layer_stack_for_testing = LayerStack(
        layers=dict(
            test=LayerLevel(
                layer=(1,0),
                thickness=1,
                zmin=0,
                material=material,
                mesh_order=1,
                background_doping_concentration=doping_conc,
                background_doping_ion=doping_type
            ),
        )
    )

    # Create test component
    component_for_testing = gf.components.rectangle(size=(1,1), layer=(1,0))
    component_for_testing.ports = dict()

    # Set temperature
    global_parameters = get_global_parameters()
    global_parameters["T"] = T

    physics_parameters = get_default_physics()
    physics_parameters[material]["mobility"] = mobility

    # Reinitialize simulation
    xsection_bounds = ((0,0.5),(1,0.5))
    simulation = DevsimComponent(
        component=component_for_testing,
        layer_stack=layer_stack_for_testing,
        mesh_type="uz",
        xsection_bounds=xsection_bounds,
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

    n_mobility = simulation.get_node_field_values(field="mu_n_node_lf", regions=["test"])[0]
    p_mobility = simulation.get_node_field_values(field="mu_p_node_lf", regions=["test"])[0]

    # Delete files to avoid clutter
    simulation.reset()

    # Return data
    return material, T, doping_conc, n_mobility, p_mobility


# ## `constant`
#
# The simplest mobility model is being constant at fixed temperature, with the temperature dependence captured by a power law:
#
# $$ \mu^{n,p} = \mu_0^{n,p} \left( \frac{T}{T_{ref}} \right)^{\alpha^{n,p}} $$
#
# with
#
# * $\mu_0$ the constant mobility at 300 K
# * $\alpha$ an exponent capturing the temperature dependence.
#
# It may be acceptable at low doping concentrations and temperatures close to 300K.
#
#

for material_name, material in materials.items():
    if material["type"] == "semiconductor" and "mobility" in material:
        if "constant" in material["mobility"]:
            print("=============================")
            print(material_name)
            print("=============================")
            mobility = material["mobility"]["constant"]
            for key, value in mobility.items():
                print(f"{key:<20} {value}")


# ## `doping_arora`
#
# Simple empirical mobility in silicon (and adapted to other semiconductors) as a function of temperature and total impurity concentration {cite}`aroraElectronHoleMobilities1982`.
#
# $$ \mu^{n,p} =  \mu_{min}^{n,p} + \frac{\mu_d^{n,p}}{1 + \left( \frac{N}{N_{ref}^{n,p}} \right)^{\alpha^{n,p}}} $$
#
# with, for both electrons (n) and holes (p):
#
# * $\mu_{min}$ is the minimum mobility value expected
# * $\mu_{d}$ is the difference between the maximum and minimum mobility expected
# * $N_{ref}$ is a reference concentration
# * $\alpha$ is an exponential factor that controls the slope around $N=N_{ref}$
#
# All four parameters are carrier type-dependent (electrons $n$, holes $p$) and are taken to be constant at a particular temperature, with temperature dependence captured by power laws:
#
# $$ \mu_{min} = A_{min} \left( \frac{T}{T_{ref}} \right)^{\alpha_m} $$
#
# $$ \mu_{d} = A_{d} \left( \frac{T}{T_{ref}} \right)^{\alpha_d} $$
#
# $$ N_{ref} = A_{N_{ref}} \left( \frac{T}{T_{ref}} \right)^{\alpha_{N_{ref}}} $$
#
# $$ \alpha = A_{\alpha} \left( \frac{T}{T_{ref}} \right)^{\alpha_\alpha} $$

def calculate_mobility(T, N, params, carrier="N"):
    """
    Calculate the mobility based on the Arora model for a given temperature and doping concentration.

    Args:
    T (float): Temperature in Kelvin.
    N (float): Doping concentration in cm^-3.
    params (dict): Dictionary containing mobility parameters for either electrons or holes.

    Returns:
    float: Calculated mobility.
    """
    mu_min = params['A_min'] * (T / params['T_ref']) ** params['alpha_m']
    mu_d = params['A_d'] * (T / params['T_ref']) ** params['alpha_d']
    N_ref = params['A_N_ref'] * (T / params['T_ref']) ** params['alpha_N_ref']
    alpha = params['A_alpha'] * (T / params['T_ref']) ** params['alpha_alpha']

    mu = mu_min + (mu_d / (1 + (N / N_ref) ** alpha))
    return mu



for material_name, material in materials.items():
    if material["type"] == "semiconductor" and "mobility" in material:
        if "doping_arora" in material["mobility"]:
            print(material_name)
            doping_arora = material["mobility"]["doping_arora"]
            for key, value in doping_arora.items():
                print(f"{key:<20} {value}")


# +
temperatures = [200, 300, 400, 500]
concentrations = np.logspace(16.5, 20, 20)

materials = ["silicon", "germanium"]
mobility_calculations = []

for temperature in temperatures:
    for concentration in concentrations:
        for material in materials:
            mobility_calculations.append(mobility_from_simulation.remote(material=material,
                                mobility=("doping_arora",),
                                T=temperature,
                                doping_conc=concentration,
                                doping_type="donor",
                                )
                            )
# -

mobilities = ray.get(mobility_calculations)
mobilities_df = pd.DataFrame(mobilities, columns=["material", "T", "doping_conc", "n_mobility", "p_mobility"])

# +
arora_silicon = pd.read_csv(Path.ref_data / "arora_silicon.csv", delimiter=",", dtype=float)
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for idx, temperature in enumerate([200, 300, 400, 500]):  # Iterate over columns for mobility values
    concentration_col = f"{temperature}_C"
    mobility_col = f"{temperature}_MU"
    color = colors[idx % len(colors)]  # Cycle through the colors list
    ax.plot(arora_silicon[concentration_col].values, arora_silicon[mobility_col].values, label=f"{temperature} K, reference", color=color)    
    
    mobility_material = mobilities_df[mobilities_df["material"] == "silicon"]
    mobility_T = mobility_material[mobility_material["T"] == temperature]
    ax.scatter(mobility_T["doping_conc"], mobility_T["n_mobility"], label=f"{temperature} K, simulator", color=color)    

ax.set_xscale('log')
ax.set_xlabel('Doping Concentration (cm^-3)')
ax.set_ylabel("Phosphorus-doped electron mobility (cm2/Vs)")
ax.set_title('Arora Model Silicon Mobility')
ax.legend()
plt.ylim([0,1000])
plt.grid(True)
plt.show()


# +
for temperature in temperatures:
    mobility_material = mobilities_df[mobilities_df["material"] == "germanium"]
    mobility_T = mobility_material[mobility_material["T"] == temperature]
    sns.scatterplot(x=np.log10(mobility_T["doping_conc"]), y=mobility_T["n_mobility"], marker='o')

sns.set(style="whitegrid")
plt.xlabel("Concentration (cm-3)")
plt.ylabel("Phosphorus-doped electron mobility (cm2/Vs)")
# plt.ylim([0,1000])
plt.legend()
# -

# <!-- # ## Bibliography
# #
# # ```{bibliography}
# # :style: unsrt
# # :filter: docname in docnames
# # ``` -->
