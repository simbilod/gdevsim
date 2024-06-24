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
# To define geometries, we use Gdsfactory Components + LayerStacks:

# +
# %load_ext autoreload
# %autoreload 2


import warnings

import matplotlib.pyplot as plt
import numpy as np

from gdevsim import ramp
from gdevsim.materials.materials import get_global_parameters
from gdevsim.samples.layers_photonic import get_layer_stack_photonic
from gdevsim.samples.optoelectronic import vertical_ge_detector
from gdevsim.simulation import DevsimComponent

warnings.filterwarnings('ignore')
# -

component = vertical_ge_detector()
component.plot(show_ports=True)

layer_stack_photonic = get_layer_stack_photonic()

# # Obtain a simulation mesh
#
# We use gplugins to mesh half of the component cross-section (since it is symmetric), keeping the layer stack labels for the moment:

# +

resolutions = {
    "core": {"resolution": 0.01, "distance": 2.0},
    "clad": {"resolution": 0.5, "distance": 1.0},
    "box": {"resolution": 0.5, "distance": 1.0},
    "ge": {"resolution": 0.01, "distance": 2.0},
}

xsection_bounds = ((10,0),(10,10))

temperature = 300
global_parameters = get_global_parameters()
global_parameters["T"] = temperature

simulation = DevsimComponent(
    component=component,
    layer_stack=layer_stack_photonic,
    mesh_type="uz",
    xsection_bounds=xsection_bounds,
    port_names_to_contact_names={"e_p1": "anode", "e_n": "cathode"},
    global_parameters=global_parameters
)
simulation.reset()

physics = {
        "silicon": {
            "bandstructure": ("bandgapnarrowing_slotboom",),
            # "mobility": ("doping_arora",), #, "highfield_canali"),
            "mobility": ("doping_arora", "highfield_canali"),
            # "generation_recombination": ("bulkSRH", "surfaceSRH"),
            "generation_recombination": ("bulkSRH",),
        },
        "germanium": {
            "bandstructure": ("bandgapnarrowing_slotboom",),
            # "mobility": ("doping_arora",), #, "highfield_canali"),
            "mobility": ("doping_arora",),
            "generation_recombination": ("bulkSRH", "surfaceSRH", "optical_generation"),
            # "generation_recombination": ("bulkSRH",),
        },
        "aluminum": {},
        "silicon_dioxide": {},
    }

simulation.initialize(
    resolutions=resolutions,
    default_characteristic_length=1.0,
    reset_save_directory=True,
    threads_available=6,
    physics_parameters=physics
)

simulation.write(file=simulation.save_directory / "test_filter.dat", type="tecplot")


# -

# simulation.plot2D(camera_dx = 0.0,
#                camera_dy = 1E-4,
#                )

# simulation.plot2D(field="AtContactNode", cmap="inferno", wireframe_color=[0,0,0])

# # Simulation

# reader = pv.get_reader(simulation.save_directory / simulation.device_data_filename)
# mesh = reader.read()
# plotter = pv.Plotter(notebook=True)
# # plotter = pv.Plotter(window_size=(1200, 1000))
# plotter.add_mesh(mesh,
#                  scalars="AtContactNode",
#                  cmap="inferno",
#                  lighting=False,
#                  show_edges=True,
#                  edge_color="grey",
#                  line_width=0.1,
#                  )
# plotter.view_xy()
# plotter.show(jupyter_backend='static')

# simulation.plot2D(field="NetDoping", field_operation=signed_log)

# +
# Apply remeshing

# steps = []
# bisections = []

# max_iterations_sequence = [10]
# solver_relative_errors_sequence = [None]
# remeshings_sequence = [refinement.default_remeshing_presolve]
# solve_flag_sequence = [False]


# for i, (max_iterations, solver_relative_errors, remeshings, solve_flag) in enumerate(zip(max_iterations_sequence, solver_relative_errors_sequence, remeshings_sequence, solve_flag_sequence)):

#     current_steps, current_bisections = simulation.remesh(remeshings=remeshings,
#                                                             default_mincl=0.0005,  # 0.5 nm
#                                                             default_maxcl=1,  # 1 um
#                                                             max_iterations=max_iterations,
#                                                             threads_available=10,
#                                                             solver_absolute_error=1e12,
#                                                             solver_relative_error=solver_relative_errors,
#                                                             solver_maximum_iterations=100,
#                                                             solve_flag=solve_flag,
#                                                             save_intermediate_data_root = f"remeshing_data_{i}_",
#                                                             save_intermediate_mesh_root = f"remeshing_mesh_{i}_",
#                                                             )

#     steps.append(current_steps)
#     bisections.append(current_bisections)



simulation.solve(threads_available=10,
    absolute_error=1e12,
    relative_error=1E-10,
    maximum_iterations=100,
    )


# # +

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
#     # plt.show()

# # -

# simulation.plot2D(field="NetDoping", field_operation=signed_log, jupyter_backend="trame")

# simulation.plot2D(field="Electrons", field_operation=signed_log, jupyter_backend="trame")

# We can now ramp the bias:

simulation.write(file = simulation.save_directory / "device_0V.ds")

simulation.write(file = simulation.save_directory / "device_0V.dat", type="tecplot")

ramp_parameters = ramp.RampParameters(contact_name="anode",
                                      biases=[-1],
                                      initial_step_size=1E-1,
                                      min_step=1E-7,
                                      rel_error=1E-12,
                                      max_iter=30,
                                      step_size_scaling_down=10,
                                      step_size_scaling_up=5,
                                      save_intermediate_structures_root="ramp")


output_reverse = simulation.ramp_dc_bias(ramp_parameters=ramp_parameters)

simulation.write(file = simulation.save_directory / "device_1V.dat", type="tecplot")


# +
plt.figure(figsize=(10, 6))

# Plotting for output_reverse
# plt.plot(output_reverse['V___anode'], np.abs(output_reverse['I___anode']), marker='o', linestyle='-', color='r', label='Reverse Anode Current vs Voltage')

# plt.title('Anode Current vs Voltage (Filtered by Bias Polarity)')
# plt.xlabel('Voltage (V)')
# plt.ylabel('Current (A)')
# plt.grid(True)
# plt.legend()
# plt.show()

# -

# # Temperature sweep
#
# Sweep the system temperature to see the impact on dark current:

# +
temperatures = np.linspace(30 + 270, 90 + 273.15, 5)
currents = []

for temperature in temperatures:
    simulation.set_parameter(name="T", value=temperature)
    simulation.solve(threads_available=10,
                     absolute_error=1e12,
                     relative_error=1E-12,
                     maximum_iterations=100,
                     )
    simulation.write(file = simulation.save_directory / f"device_{temperature}.dat", type="tecplot")
    currents.append(simulation.get_current(contact="anode"))

# -

# Scale currents in A/cm to A/um, x100 um length, scale A to uA, and then x2 from symmetry
plt.plot(temperatures - 273.15, np.abs(currents) * 1E-4 * 100 * 1E6 * 2)
plt.xlabel("Temperature (C)")
plt.ylabel("Dark current (uA)")
plt.yscale('log')
plt.show()
