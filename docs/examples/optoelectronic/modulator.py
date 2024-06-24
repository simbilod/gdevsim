# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: devsim
#     language: python
#     name: python3
# ---

# # Define geometry
#
# We define the GDSFactory Component + LayerStack:

# +
# %load_ext autoreload
# %autoreload 2

from pprint import pprint

import gdsfactory as gf
import matplotlib.pyplot as plt
import meshio
import numpy as np
import yaml
from femwell.maxwell.waveguide import compute_modes
from femwell.pn_analytical import (
    alpha_to_k,
    dalpha_carriers,
    dn_carriers,
)
from femwell.visualization import plot_domains
from gdsfactory.cross_section import pn
from scipy.interpolate import LinearNDInterpolator
from skfem import Basis, ElementTriP0
from skfem.io.meshio import from_meshio

from gdevsim import ramp
from gdevsim.meshing import refinement
from gdevsim.samples.layers_photonic import LAYER, get_layer_stack_photonic
from gdevsim.samples.optoelectronic import (
    straight_pn,
    straight_pn_via_ports,
    via_stack,
    viac,
)
from gdevsim.simulation import DevsimComponent
from gdevsim.utils.operations import signed_log

gf.config.CONF.logger.disable = True


# +
# Edit the Component geometry
xs_pn = pn(
            width = 0.5, # core width
            layer = "WG", # core GDS layer
            layer_slab = "SLAB90", # slab GDS layer
            gap_low_doping = 0.0,
            gap_medium_doping = 0.5,
            gap_high_doping = 1.0,
            offset_low_doping = 0.0,
            width_doping = 8.0,
            width_slab = 7.0,
            layer_p = "P",
            layer_pp = "PP",
            layer_ppp = "PPP",
            layer_n = "N",
            layer_np = "NP",
            layer_npp = "NPP",
         )

straight_pn_simulation = straight_pn_via_ports(straight_pn(length=15,
                                       cross_section=xs_pn,
                                        via_stack=gf.partial(via_stack,
                                                                layers = (None, LAYER.M1,),
                                                                vias = (viac,),
                                                            ),
                                        via_stack_width=2,
                                        taper=None)
                                        )

straight_pn_simulation.plot(show_ports=True)

# +
layer_stack_photonic = get_layer_stack_photonic()

# Edit the LayerStack parameters
layer_stack_photonic.layers["n"].info["peak_concentrations"] = (5E17,)
layer_stack_photonic.layers["p"].info["peak_concentrations"] = (5E17,)

pprint(layer_stack_photonic.to_dict())
# -

# ## Define simulation
#
# We initialize the simulation:

# +
resolutions = {
    "slab": {"resolution": 0.05, "distance": 1.0},
    "core": {"resolution": 0.02, "distance": 1.0},
    "clad": {"resolution": 0.5, "distance": 1.0},
    "box": {"resolution": 0.5, "distance": 1.0},
}

xsection_bounds = ((15 / 2, -5), (15 / 2, 5))


simulation = DevsimComponent(
    component=straight_pn_simulation,
    layer_stack=layer_stack_photonic,
    mesh_type="uz",
    xsection_bounds=xsection_bounds,
    port_names_to_contact_names={"e_p": "anode", "e_n": "cathode"},
)
simulation.reset()

physics = {
        "silicon": {
            "bandstructure": ("bandgapnarrowing_slotboom",),
            "mobility": ("doping_arora",),
            "generation_recombination": ("bulkSRH",),
            # "generation_recombination": ("bulkSRH",),
        },
        "aluminum": {},
        "silicon_dioxide": {},
    }

output_filename = simulation.initialize(
    resolutions=resolutions,
    default_characteristic_length=1.0,
    reset_save_directory=True,
    threads_available=6,
    physics_parameters=physics
)
# -

# We can inspect the simulation:

# simulation.plot2D(camera_dx = 0.0,
#                camera_dy = 1E-4,
#                )

# simulation.plot2D(field="AtContactNode", cmap="inferno")

# simulation.plot2D(field="NetDoping", field_operation=signed_log)
# ## Adaptive remeshing
#
# While we could use the device as-is, it is good practice to refine the mesh in regions where fields change quickly. This is easily achieved with remeshing operations:
# +
# Apply remeshing

steps = []
bisections = []

max_iterations_sequence = [2, 3]
solver_relative_errors_sequence = [1E-10, 1E-12]
remeshings_sequence = [refinement.default_remeshing_postsolve, refinement.default_remeshing_postsolve]
solve_flag_sequence = [True, True]

simulation.solve(threads_available=10,
    absolute_error=1e12,
    relative_error=1E-10,
    maximum_iterations=100,
    )

for i, (max_iterations, solver_relative_errors, remeshings, solve_flag) in enumerate(zip(max_iterations_sequence, solver_relative_errors_sequence, remeshings_sequence, solve_flag_sequence)):

    current_steps, current_bisections = simulation.remesh(remeshings=remeshings,
                                                            default_mincl=0.0005,  # 0.5 nm
                                                            default_maxcl=1,  # 1 um
                                                            max_iterations=max_iterations,
                                                            threads_available=10,
                                                            solver_absolute_error=1e12,
                                                            solver_relative_error=solver_relative_errors,
                                                            solver_maximum_iterations=100,
                                                            solve_flag=solve_flag,
                                                            save_intermediate_data_root = f"remeshing_data_{i}_",
                                                            save_intermediate_mesh_root = f"remeshing_mesh_{i}_",
                                                            )

    steps.append(current_steps)
    bisections.append(current_bisections)

# +

# # Prepare data for seaborn
# for i in range(len(remeshings_sequence)):
#     data = []
#     bisection_i = bisections[i]
#     steps_i = steps[i]
#     for region, bisection_counts in bisection_i.items():
#         for step, count in zip(steps_i, bisection_counts):
#             data.append({'Remeshing step': step, 'Bisections': count, 'Region': region})
#     df = pd.DataFrame(data)

#     # Plot with seaborn
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(10, 6))
#     plt.title(f"Remeshing stage {i}")
#     sns.lineplot(data=df, x="Remeshing step", y="Bisections", hue="Region", linewidth=2.5)
#     plt.xlabel("Remeshing step", fontsize=14)
#     plt.ylabel("Bisections", fontsize=14)
#     plt.legend(title="Region", fontsize=12)
#     plt.show()

# -

# simulation.plot2D(field="NetDoping", field_operation=signed_log, jupyter_backend="trame")

# simulation.plot2D(field="Electrons", field_operation=signed_log, jupyter_backend="trame")

# # Ramping DC bias

# First, we will checkpoint the initialized device:

simulation.write(file = simulation.save_directory / "device_0V.ds")

# We can now ramp the bias:
ramp_parameters = ramp.RampParameters(contact_name="anode", biases=np.linspace(0, 1, 11), save_intermediate_structures_root="ramp")

simulation.load(file = simulation.save_directory / "device_0V.ds")
output_forward = simulation.ramp_dc_bias(ramp_parameters=ramp_parameters)

ramp_parameters = ramp.RampParameters(contact_name="anode", biases=np.linspace(0, -5, 6), save_intermediate_structures_root="ramp")
simulation.load(file = simulation.save_directory / "device_0V.ds")
output_reverse = simulation.ramp_dc_bias(ramp_parameters=ramp_parameters)

# +
plt.figure(figsize=(10, 6))

# Plotting for output_forward
plt.semilogy(output_forward['V___anode'], np.abs(output_forward['I___anode']), marker='o', linestyle='-', color='g', label='Forward Anode Current vs Voltage')

# Plotting for output_reverse
plt.semilogy(output_reverse['V___anode'], np.abs(output_reverse['I___anode']), marker='o', linestyle='-', color='r', label='Reverse Anode Current vs Voltage')

plt.title('Anode Current vs Voltage (Filtered by Bias Polarity)')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.ylim([1E-16, 1E4])
plt.grid(True)
plt.legend()
plt.show()

# -

# # Inspecting the charge densities

# We can inspect the charge densities:

# +
files_with_voltage = []
for f in simulation.save_directory.iterdir():
    if (f.name.startswith("ramp") and f.name.endswith(".ds")):
        with open(f.with_suffix(".yaml")) as file:
            data = yaml.safe_load(file)
        files_with_voltage.append((f, float(data[simulation.device_name]["slab___via@e_p_bias"])))

# Also get the 1D cuts at middle of core
x = simulation.get_node_field_values(field="x", regions=["core"])
y = simulation.get_node_field_values(field="y", regions=["core"])
xmin, xmax = np.min(x), np.max(x)
xs = np.linspace(xmin, xmax, 101)
y_mid = np.mean(y)
electrons_cut = {}
holes_cut = {}
positions = np.array([(x, y_mid) for x in xs])

sorted_files = sorted(files_with_voltage, key=lambda x: x[1])
for _i, (file, voltage) in enumerate(sorted_files[:10]):
    simulation.load(file=file)
    simulation.plot2D(field="Electrons",
                        field_operation=signed_log,
                        jupyter_backend='static',
                        show_wireframe=None,
                        zoom=20,
                        camera_dy=0.9E-4,
                        title=f"Voltage = {voltage:1.3f} V",
                        )

    electrons_cut[voltage] = simulation.probe_field_values(positions=positions, field="Electrons")
    holes_cut[voltage] = simulation.probe_field_values(positions=positions, field="Holes")
# -

cmap = plt.get_cmap("viridis")
colors = cmap(np.linspace(0, 1, len(electrons_cut)))
for _i, (voltage, color) in enumerate(zip(electrons_cut.keys(), colors)):
    plt.plot(xs, holes_cut[voltage], linestyle="--", color=color)
    plt.plot(xs, electrons_cut[voltage], linestyle="-", color=color, label=f"{voltage:1.1f} V")
plt.xlabel("x-position (cm)")
plt.ylabel("Concentration (cm-3)")
plt.legend(title="Voltage", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title("y = 110 nm cut")

# # Calculating phase shift
#
# We can solve for the profile of the mode supported by this waveguide, and project the charge distribution as an index distribution to calculate phase shift.
#
# Since `femwell` can directly use the simulation mesh, it is easy to get the unperturbed mode:

# +

filepath = simulation.save_directory / simulation.mesh_filename # use the initial mesh
mesh = meshio.read(filepath, file_format='gmsh')
mesh = from_meshio(mesh)
mesh.draw().show()
# -

plot_domains(mesh)
plt.show()

basis0 = Basis(mesh, ElementTriP0())
epsilon = basis0.zeros()
for subdomain, n in {"core": 3.45,
                     "slab": 3.45,
                     "box": 1.444,
                     "clad": 1.444,
                     "via@e_n": 1.444, # ignore the contacts
                     "via@e_p": 1.444, # ignore the contacts
                     }.items():
    epsilon[basis0.get_dofs(elements=subdomain)] = n**2
basis0.plot(epsilon, colorbar=True).show()

# +
wavelength = 1.55 * 1E-4 # in cm

modes = compute_modes(basis0, epsilon, wavelength=wavelength, num_modes=1, order=2)
mode = modes[0]
# -

neff_unperturbed = mode.n_eff

mode.plot(field=np.real(mode.E), plot_vectors=False, colorbar=True, direction="y", title="E")
plt.xlim([4E-4,6E-4])
plt.ylim([-1E-5,4E-5])
plt.title("Normalized Intensity")
plt.tight_layout()

# We can now perturb the mode with the calculated carrier distribution, converted to refractive index via the Soref Equations for free carrier dispersion in silicon. Visualizing with the 0V distribution:

simulation.load(file = simulation.save_directory / "device_0V.ds")

# +

x = simulation.get_node_field_values(field="x", regions=["core","clad"])
y = simulation.get_node_field_values(field="y", regions=["core","clad"])
electrons = simulation.get_node_field_values(field="Electrons", regions=["core","clad"])
holes = simulation.get_node_field_values(field="Holes", regions=["core","clad"])
dn = dn_carriers(wavelength=1.55, dN=electrons, dP=holes)
dk = alpha_to_k(dalpha_carriers(wavelength=1.55, dN=electrons, dP=holes), wavelength=1.55)

X = np.sort(np.unique(x))
Y = np.sort(np.unique(y))
XX, YY = np.meshgrid(X, Y)

dn_interp = LinearNDInterpolator(list(zip(x, y)), dn)
dk_interp = LinearNDInterpolator(list(zip(x, y)), dk)

dn_df = dn_interp(XX, YY)
dk_df = dk_interp(XX, YY)

dn_Si = np.nan_to_num(modes[0].basis_epsilon_r.project(
        lambda x: dn_interp(x[0], x[1]) -1j*dk_interp(x[0], x[1]),
        dtype=complex,
        )
    )
# -

dn = modes[0].basis_epsilon_r.zeros(dtype=complex)
dn_Si = np.nan_to_num(modes[0].basis_epsilon_r.project(
        lambda x: dn_interp(x[0], x[1]) -1j*dk_interp(x[0], x[1]),
        dtype=complex,
        )
    )
dn_Si[mode.basis_epsilon_r.get_dofs(elements=list(set(mode.basis_epsilon_r.mesh.subdomains.keys()) - {"core", "slab"}))] = 0.0
dn += dn_Si

epsilon = mode.basis_epsilon_r.zeros(dtype=complex)
for subdomain, n in {"core": 3.45,
                     "slab": 3.45,
                     "box": 1.444,
                     "clad": 1.444,
                     "via@e_n": 1.444,
                     "via@e_p": 1.444,
                     }.items():
    dofs = mode.basis_epsilon_r.get_dofs(elements=subdomain)
    epsilon[mode.basis_epsilon_r.get_dofs(elements=subdomain)] = (n + dn[dofs])**2
basis0.plot(np.imag(epsilon), colorbar=True)
plt.xlim([4E-4,6E-4])
plt.ylim([-0.05E-4, 0.5E-4])
plt.show()

basis0.plot(np.log10(np.real(epsilon)), colorbar=True)
plt.xlim([4E-4,6E-4])
plt.ylim([-0.05E-4, 0.5E-4])
plt.show()

neff_p = mode.calculate_pertubated_neff(epsilon - mode.epsilon_r)

neff_p

# Now we consider the effective index as a function of the applied voltage:

# +
neffs = {}

for _i, (file, voltage) in enumerate(sorted_files[:10]):
    simulation.load(file=file)
    x = simulation.get_node_field_values(field="x", regions=["core","clad"])
    y = simulation.get_node_field_values(field="y", regions=["core","clad"])
    electrons = simulation.get_node_field_values(field="Electrons", regions=["core","clad"])
    holes = simulation.get_node_field_values(field="Holes", regions=["core","clad"])
    dn = dn_carriers(wavelength=1.55, dN=electrons, dP=holes)
    dk = alpha_to_k(dalpha_carriers(wavelength=1.55, dN=electrons, dP=holes), wavelength=1.55)

    X = np.sort(np.unique(x))
    Y = np.sort(np.unique(y))
    XX, YY = np.meshgrid(X, Y)

    dn_interp = LinearNDInterpolator(list(zip(x, y)), dn)
    dk_interp = LinearNDInterpolator(list(zip(x, y)), dk)

    dn_df = dn_interp(XX, YY)
    dk_df = dk_interp(XX, YY)

    dn_Si = np.nan_to_num(modes[0].basis_epsilon_r.project(
            lambda x: dn_interp(x[0], x[1]) -1j*dk_interp(x[0], x[1]), # noqa: B023
            dtype=complex,
            )
        )

    dn = modes[0].basis_epsilon_r.zeros(dtype=complex)
    dn_Si = np.nan_to_num(modes[0].basis_epsilon_r.project(
            lambda x: dn_interp(x[0], x[1]) -1j*dk_interp(x[0], x[1]), # noqa: B023
            dtype=complex,
            )
        )
    dn_Si[mode.basis_epsilon_r.get_dofs(elements=list(set(mode.basis_epsilon_r.mesh.subdomains.keys()) - {"core", "slab"}))] = 0.0
    dn += dn_Si

    epsilon = mode.basis_epsilon_r.zeros(dtype=complex)
    for subdomain, n in {"core": 3.45,
                        "slab": 3.45,
                        "box": 1.444,
                        "clad": 1.444,
                        "via@e_n": 1.444,
                        "via@e_p": 1.444,
                        }.items():
        dofs = mode.basis_epsilon_r.get_dofs(elements=subdomain)
        epsilon[mode.basis_epsilon_r.get_dofs(elements=subdomain)] = (n + dn[dofs])**2

    neffs[voltage] = mode.calculate_pertubated_neff(epsilon - mode.epsilon_r)
# -

# plt.plot(neffs_df["voltage"], np.real(neffs_df["neff"]))
# plt.xlabel("Voltage (V)")
# plt.ylabel("Effective index shift (a.u.)")

# plt.plot(neffs_df["voltage"], k_to_alpha_dB(np.imag(neffs_df["neff"]), wavelength=1.55))
# plt.xlabel("Voltage (V)")
# plt.ylabel("Absorption (dB/cm)")
