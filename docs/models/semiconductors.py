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

# # Semiconductors
#
# Semiconductors are defined as materials with a bandgap ~ energy scales of the problem. Hence, they exhibit variable concentrations of mobile charge, greatly affecting conduction. This is mainly determined by the introduction of impurity atoms having more (donors) or less (acceptors) outer shell electrons. Furthermore, various mechanisms can act as sinks (recombination $R$) or sources (generation $G$) of mobile charge.
#
# * **Parameters**: 
#     * Acceptor concentration $N_A(\bm{x})$ (/cm3)
#     * Donor concentration $N_D(\bm{x})$ (/cm3)
#     * Band models + parameters
#     * Mobility models + parameters $\mu_e(...), \mu_h(...)$
#     * Recombination-generation models + parameters $R(...), G(...)$
# * **Node solutions**: 
#     * Electrostatic potential $\varphi (\bm{x})$ (V)
#     * Electron concentration $n_e(\bm{x})$ (/cm3)
#     * Hole concentration $n_h(\bm{x})$ (/cm3)
# * **Equations**
#     * Electron drift current $\bm{J}_e = q n_e \mu_e \nabla V$
#     * Electron diffusion current $\bm{J}_e = q n_e \mu_e \nabla n_e(x)$
#     * Hole drift current $\bm{J}_h = q n_h \mu_h \nabla V$
#     * Coulomb' law: $\epsilon_r \nabla^2 \varphi = q \left( n_h(x) - n_e(x) + N_D(x) - N_A(x) \right)$
#     * Electron continuity: 
#     * Hole continuity: 
#
# We instanciate generic simulations to benchmark the parameters:

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
from gdevsim.materials.materials import get_global_parameters
all_materials = get_all_materials()

ray.init(log_to_driver=False)


# -

@ray.remote
def carriers_from_simulation(material: str, 
                        T: float, 
                        doping_conc: float, 
                        doping_type: str,
                        ):
    """Test object to check how carrier models is implemented in simulations.

    Arguments:
        material: material name (str)
        carrier: carrier models to use
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

    # Reinitialize simulation
    xsection_bounds = ((0,0.5),(1,0.5))
    simulation = DevsimComponent(
        component=component_for_testing,
        layer_stack=layer_stack_for_testing,
        mesh_type="uz",
        xsection_bounds=xsection_bounds,
        global_parameters=global_parameters,
    )
    simulation.reset()

    # Read instanciated data
    simulation.initialize(
        default_characteristic_length=1.0,
        reset_save_directory=True,
        threads_available=1,
    )
    data = {}
    data["material"] = material
    data["T"] = T
    data["doping_conc"] = doping_conc
    data["doping_type"] = doping_type
    for name in ["EG", "DEG", "NC", "NV", "NIE", "EC", "EV", "EI", ]:
        data[name] = simulation.get_node_field_values(field=name, regions=["test"])[0]
    for name in ["NC300", "NV300", "EG300", "EGALPH", "EGBETA", "Affinity"]:
        data[name] = simulation.get_parameter(parameter=name, region="test")

    # Delete files to avoid clutter
    simulation.reset()

    # Return data
    return data


# +
temperatures = np.arange(50, 501, 50)
concentrations = [0] + list(np.logspace(16.5, 20, 20))

semiconducting_materials = ["silicon", "germanium"]
carrier_calculations = []

for temperature in temperatures:
    for concentration in concentrations:
        for material in semiconducting_materials:
            carrier_calculations.append(carriers_from_simulation.remote(material=material,
                                T=float(temperature),
                                doping_conc=concentration,
                                doping_type="donor",
                                )
                            )

# +
carrier_data = ray.get(carrier_calculations)
import pandas as pd

# Create a DataFrame from the list of dictionaries
carrier_df = pd.DataFrame(carrier_data)
# -

# ## Bandgap
#
# Bandgap and its temperature dependence is captured as
#
# $$ E_G = E_{G0} + \alpha \left( \frac{T_0^2}{T_0 + \beta} - \frac{T^2}{T + \beta} \right) $$
#
# where $T_{0} = 300K$ is a reference temperature, and $E_{G0}$ the bandgap at the reference temperature. 

# +
import pandas as pd

# Create an empty list to store the data
bandstructure_data = []

for material_name, material in all_materials.items():
    if material["type"] == "semiconductor":
        data = material["bandstructure"]
        for key, value in data.items():
            if not isinstance(value, dict):
                bandstructure_data.append({
                    "Material": material_name,
                    "Property": key,
                    "Reference value": float(value),
                    "Simulator value": float(carrier_df[(carrier_df['material'] == material_name) & (carrier_df['doping_conc'] == 0 ) & (carrier_df['T'] == 300 )][key].values[0]),
                })

# Convert the list of dictionaries to a DataFrame
bandstructure_df = pd.DataFrame(bandstructure_data)

# Apply style to the DataFrame for better visualization
display(bandstructure_df.style.format(lambda x: "{:.3e}".format(x) if not isinstance(x, str) else x).set_properties(**{'background-color': 'black', 'color': 'white', 'border-color': 'gray'}))


# +
def ref_bandgap(T, material):
    if material == "silicon":
        # Silicon
        # http://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html
        return 1.17 - 4.73 * 1E-4 * T**2/(T+636)
    elif material == "germanium":
        # Germanium
        # https://www.ioffe.ru/SVA/NSM/Semicond/Ge/bandstr.html
        return 0.742 - 4.8 * 1E-4 * T**2/(T+235)
    else:
        raise ValueError(f"Material {material} has no reference!")


temperature_range = np.linspace(0, 500, 500)

for material_name, material in all_materials.items():
    if material["type"] == "semiconductor":
        # Calculate the reference bandgap for each temperature
        bandgaps = [ref_bandgap(T, material_name) for T in temperature_range]
        # Filter the DataFrame for silicon material
        data = carrier_df[(carrier_df['material'] == material_name) & (carrier_df['doping_conc'] == 0 )]
        # Plot the EG field for silicon and the reference bandgap on the same plot
        plt.figure(figsize=(8, 5))
        plt.plot(temperature_range, bandgaps, 'r-', label='reference')
        plt.scatter(data['T'], data['EG'], color='b', label='simulation')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Bandgap (eV), for $\Delta E_G = 0$')
        plt.title(material_name)
        plt.legend()
        plt.grid(True)
        plt.show()

# -

# ## Bandgap narrowing
#
# Bandgap narrowing from the formation of impurity bands is captured as
#
# $$ \Delta E_G = E_{BGN} * \left(\log\left(\frac{N_{tot}}{N_{ref}} \right) + \sqrt{\left( \log\left( \frac{N_{tot}}{N_{ref}} \right)\right)^2 + C_{BGN}} \right) $$
#
# where $N_{tot} = N_A + N_D$ is the total local concentration of impurities. 
#
# Compare to data compiled in 
#
# [1] R. J. Van Overstraeten and R. P. Mertens, “Heavy doping effects in silicon,” Solid-State Electronics, vol. 30, no. 11, pp. 1077–1087, Nov. 1987, doi: 10.1016/0038-1101(87)90070-0.
#
# [2] S. C. Jain and D. J. Roulston, “A simple expression for band gap narrowing (BGN) in heavily doped Si, Ge, GaAs and GexSi1−x strained layers,” Solid-State Electronics, vol. 34, no. 5, pp. 453–465, May 1991, doi: 10.1016/0038-1101(91)90149-S.

for material_name, material in all_materials.items():
    if material["type"] == "semiconductor":
        print("===============================")
        print(material_name)
        print("===============================")
        data = material["bandstructure"]
        for key, value in data["bandgapnarrowing_slotboom"].items():
            print(f"{key:<20} {float(value):.3e}")

# +
BGN_data = {}
BGN_data["silicon"] = pd.read_csv(Path.ref_data / "slotboom_silicon.csv", delimiter=";", decimal=",")

def ref_bgn(c, material):
    if material == "silicon":
        # Silicon
        E0BGN = 9.000e-03
        N0BGN = 1.000e+17
        CONBGN = 5.000e-01
        if c == 0:
            return 0
        else:
            return E0BGN * ( np.log(c/N0BGN) + np.sqrt( (np.log(c/N0BGN))**2 + CONBGN ) )
    elif material == "germanium":
        # Silicon
        E0BGN = 9.000e-03
        N0BGN = 1.000e+17
        CONBGN = 5.000e-01
        if c == 0:
            return 0
        else:
            return E0BGN * ( np.log(c/N0BGN) + np.sqrt( (np.log(c/N0BGN))**2 + CONBGN ) )
    else:
        raise ValueError(f"Material {material} has no reference!")
    

for material_name, material in all_materials.items():
    if material["type"] == "semiconductor":
        bgns =  [ref_bgn(c, material_name) * 1000 for c in concentrations]
        data = carrier_df[(carrier_df["material"] == material_name) & (carrier_df['T'] == 300 )]
        plt.figure(figsize=(8, 5))
        plt.plot(concentrations, bgns, 'r-', label='reference')
        plt.scatter(data['doping_conc'], data['DEG'], color='b', label='simulation')
        # If data, add
        # if material_name in BGN_data:
        #     data_x = np.log10(np.array(BGN_data[material_name]["ND (cm-3)"].values, dtype=float))
        #     data_y = np.array(BGN_data[material_name]["DEG (meV)"].values, dtype=float)
        #     plt.plot(data_x, data_y, label="data")
        plt.xlabel('Donor Concentration (cm-3)')
        plt.ylabel('Bandgap narrowing (meV), for $T = 300$')
        plt.title(material_name)
        plt.legend()
        plt.grid(True)
        plt.show()


# -

# ## Effective densities of state
#
# $$ N_C = N_C^0 \left(\frac{T}{T_0} \right)^{\frac{3}{2}} $$
#
# $$ N_V = N_V^0 \left(\frac{T}{T_0} \right)^{\frac{3}{2}} $$

# +
def ref_DOS_V(T, material):
    if material == "silicon":
        # Silicon
        # http://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html
        return 3.5 * 1E15 * T**(3/2)
    elif material == "germanium":
        # Germanium
        # https://www.ioffe.ru/SVA/NSM/Semicond/Ge/bandstr.html
        return 9.6 * 1E14 * T**(3/2)
    else:
        raise ValueError(f"Material {material} has no reference!")
    
def ref_DOS_C(T, material):
    if material == "silicon":
        # Silicon
        # http://www.ioffe.ru/SVA/NSM/Semicond/Si/bandstr.html
        return 6.2 * 1E15 * T**(3/2)
    elif material == "germanium":
        # Germanium
        # https://www.ioffe.ru/SVA/NSM/Semicond/Ge/bandstr.html
        return 1.98 * 1E15 * T**(3/2)
    else:
        raise ValueError(f"Material {material} has no reference!")


# -

temperature_range = np.linspace(0, 500, 500)
for material_name, material in all_materials.items():
    if material["type"] == "semiconductor":
        # Calculate the reference bandgap for each temperature
        DOS_V = [ref_DOS_V(T, material_name) for T in temperature_range]
        DOS_C = [ref_DOS_C(T, material_name) for T in temperature_range]
        # Filter the DataFrame for silicon material
        data = carrier_df[carrier_df['material'] == material_name]
        # Plot the EG field for silicon and the reference bandgap on the same plot
        plt.figure(figsize=(8, 5))
        plt.plot(temperature_range, DOS_C, 'b-', label='reference NC')
        plt.plot(temperature_range, DOS_V, 'g-', label='reference NV')
        plt.scatter(data['T'], data['NC'], color='b', label='simulation NC')
        plt.scatter(data['T'], data['NV'], color='g', label='simulation NV')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Density of states (cm-3)')
        plt.title(material_name)
        plt.legend()
        plt.grid(True)
        plt.show()


# ## Intrinsic carrier concentration

# The intrinsic carrier concentration should be calculated from the above as
#
# $$ n_i = \sqrt{ N_C N_V } e^{\frac{-E_G}{2k_BT}} $$

# +
temperature_range = np.linspace(0, 500, 500)

def ni(T, EG, NC, NV):
    kB = 8.617e-5
    return np.sqrt(NC * NV) * np.exp(-EG / (2*kB*T))

for material_name, material in all_materials.items():
    if material["type"] == "semiconductor":
        bandgaps = np.array([ref_bandgap(T, material_name) for T in temperature_range])
        DOS_C = np.array([ref_DOS_C(T, material_name) for T in temperature_range])
        DOS_V = np.array([ref_DOS_V(T, material_name) for T in temperature_range])
        # Filter the DataFrame for silicon material
        data = carrier_df[(carrier_df['material'] == material_name) & (carrier_df['doping_conc'] == 0 )]
        # Plot the EG field for silicon and the reference bandgap on the same plot
        plt.figure(figsize=(8, 5))
        plt.semilogy(temperature_range, ni(temperature_range, bandgaps, DOS_C, DOS_V), 'r-', label='reference')
        plt.semilogy(data['T'], data['NIE'], color='b', marker='o', linestyle='None', label='simulation')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Intrinsic carrier concentration (cm-3)')
        plt.title(material_name)
        plt.ylim([1E-40,1E20])
        plt.legend()
        plt.grid(True)
        plt.show()
