"""Adapted from DEVSIM LLC:
    - https://github.com/devsim/devsim/blob/main/python_packages/simple_physics.py
    - https://github.com/devsim/devsim_bjt_example/blob/main/simdir/physics/model_create.py

Improvements:
    - more Pythonic names
    - cleaner imports
    - f-strings
"""

import devsim as ds

from gdevsim.models.create_models import (
    create_arithmetic_mean,
    create_arithmetic_mean_derivative,
    create_contact_node_model,
    create_edge_model,
    create_edge_model_derivatives,
    create_interface_model,
    create_node_model,
    create_node_model_derivative,
    create_solution,
    ensure_edge_from_node_model_exists,
    in_edge_model_list,
    in_node_model_list,
)

"""
COMMON
"""


def get_contact_bias_name(contact):
    return f"{contact}_bias"


def GetContactNodeModelName(contact):
    return f"{contact}nodemodel"


def create_material_indexing(device, region, interface_delimiter="___"):
    """Create indexing functions to tag interior and surfaces of regions.

    Interfaces of two regions of the same material are tagged as bulk.

    Convention:
        {region}_at_bulk: 1 for interior nodes of region, 0 otherwise
        {region}_at_interface_{interface}: 1 for nodes of region touching interface, 0 otherwise
    """
    current_interfaces = [
        interface
        for interface in ds.get_interface_list(device=device)
        if region in interface
    ]
    # Tag interfaces by materials
    at_bulk_expression = "1"
    for interface in current_interfaces:
        region0, region1 = interface.split(interface_delimiter)
        material0, material1 = ds.get_material(
            device=device, region=region0
        ), ds.get_material(device=device, region=region1)
        if not material0 == material1:
            at_interface_name = f"region_{region}_at_interface_{interface}"
            ds.node_solution(device=device, region=region, name=at_interface_name)
            elist = ds.get_element_node_list(
                device=device, interface=interface, region=region
            )
            nset = set()
            for e in elist:
                nset.update(e)
            for n in nset:
                ds.set_node_value(
                    device=device,
                    region=region,
                    name=at_interface_name,
                    index=n,
                    value=1.0,
                )
            at_bulk_expression += f" * (1 - region_{region}_at_interface_{interface})"

    create_node_model(device, region, f"region_{region}_at_bulk", at_bulk_expression)


def create_efield(device, region):
    """
    Creates the electrical field from the gradient of the potential.
    """
    ds.edge_average_model(
        device=device,
        region=region,
        node_model="Potential",
        edge_model="EField",
        average_type="negative_gradient",
    )
    ds.edge_average_model(
        device=device,
        region=region,
        node_model="Potential",
        edge_model="EField",
        average_type="negative_gradient",
        derivative="Potential",
    )


def create_dfield(device, region):
    create_edge_model(device, region, "DField", "eps_0 * eps_r * EField")
    # Derivatives
    create_edge_model(
        device, region, "DField:Potential@n0", "eps_0 * eps_r * EField:Potential@n0"
    )
    create_edge_model(
        device, region, "DField:Potential@n1", "eps_0 * eps_r * EField:Potential@n1"
    )


"""
INSULATORS
"""


def create_insulator_potential(device, region, update_type="default"):
    """
    Create electric field model in insulator
    Creates Potential solution variable if not available
    """
    if not in_node_model_list(device, region, "Potential"):
        print("Creating Node Solution Potential")
        create_solution(device, region, "Potential")

    create_efield(device, region)
    create_dfield(device, region)

    ds.equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        edge_model="DField",
        variable_update=update_type,
    )


def create_insulator_potentialOnly_contact(device, region, contact):
    """
    Creates the potential equation at the semiconductor contact
    """
    if not in_edge_model_list(device, region, "contactcharge_edge"):
        create_edge_model(device, region, "contactcharge_edge", "eps_0*eps_r*EField")
        create_edge_model_derivatives(
            device,
            region,
            "contactcharge_edge",
            "eps_0*eps_r*EField",
            "Potential",
        )

    contact_model = f"Potential -{get_contact_bias_name(contact)}"

    contact_model_name = GetContactNodeModelName(contact)
    create_contact_node_model(device, contact, contact_model_name, contact_model)
    create_contact_node_model(
        device, contact, "{}:{}".format(contact_model_name, "Potential"), "1"
    )

    ds.contact_equation(
        device=device,
        contact=contact,
        name="PotentialEquation",
        node_model=contact_model_name,
        edge_model="",
        node_charge_model="",
        edge_charge_model="contactcharge_edge",
        node_current_model="",
        edge_current_model="",
    )


"""
SEMICONDUCTORS
"""


def create_vt(device, region, variables):
    """
    Calculates the thermal voltage, based on the temperature.
    V_t : node model
    V_t_edge : edge model from arithmetic mean
    """
    create_node_model(device, region, "V_t", "k*T/q")
    create_arithmetic_mean(device, region, "V_t", "V_t_edge")
    if "T" in variables:
        create_arithmetic_mean_derivative(device, region, "V_t", "V_t_edge", "T")


def create_quasi_fermi_levels(device, region, electron_model, hole_model, variables):
    """
    Creates the models for the quasi-Fermi levels.  Assuming Boltzmann statistics.
    """
    eq = (
        ("EFN", f"EC + V_t * log({electron_model}/NC)", ("Potential", "Electrons")),
        ("EFP", f"EV - V_t * log({hole_model}/NV)", ("Potential", "Holes")),
    )
    for model, equation, variable_list in eq:
        create_node_model(device, region, model, equation)
        vset = set(variable_list)
        for v in variables:
            if v in vset:
                create_node_model_derivative(device, region, model, equation, v)



def create_density_of_states(device, region, physics, variables):
    """
    Set up models for density of states.
    """
    eq = (
        ("NC", "NC300 * (T/300)^1.5", ("T",)),
        ("NV", "NV300 * (T/300)^1.5", ("T",)),
        # We set the total impurity to a small nonzero number to prevent DEG from crashing
        ("NTOT", "ifelse( Donors + Acceptors > 1, Donors + Acceptors, 1)", ()),
        # Band Gap Narrowing
        ('DEG', 'V0.BGN * (log(NTOT/N0.BGN) + ((log(NTOT/N0.BGN)^2 + CON.BGN)^(0.5)))' if "bandgapnarrowing_slotboom" in physics else "0", ()),
        # ('DEG', 'if( NTOT >= 1E20, V0BGN * (log(NTOT/N0BGN) + ((log(NTOT/N0BGN)^2 + CONBGN)^(0.5))))' if "bandgapnarrowing_slotboom" in physics else "0", ()),
        ("EG", "EG300 + EGALPH*((300^2)/(300+EGBETA) - (T^2)/(T+EGBETA)) - DEG", ("T")),
        ("NIE", "((NC * NV)^0.5) * exp(-EG/(2*V_t))*exp(DEG)", ("T")),
        ("EC", "-Potential - Affinity - DEG/2", ("Potential",)),
        ("EV", "EC - EG + DEG/2", ("Potential", "T")),
        ("EI", "0.5 * (EC + EV + V_t*log(NC/NV))", ("Potential", "T")),
    )

    for model, equation, variable_list in eq:
        create_node_model(device, region, model, equation)
        vset = set(variable_list)
        for v in variables:
            if v in vset:
                create_node_model_derivative(device, region, model, equation, v)


def create_semiconductor_potential(device, region, physics):
    """
    Creates the physical models for a Silicon region for equilibrium simulation.
    """
    if not in_node_model_list(device, region, "Potential"):
        print("Creating Node Solution Potential")
        create_solution(device, region, "Potential")

    variables = ("Potential",)
    create_vt(device, region, variables)
    create_density_of_states(device, region, physics["bandstructure"], variables)

    # require NetDoping
    for i in (
        ("IntrinsicElectrons", "NIE*exp(Potential/V_t)"),
        ("IntrinsicHoles", "NIE^2/IntrinsicElectrons"),
        ("IntrinsicCharge", "kahan3(IntrinsicHoles, -IntrinsicElectrons, NetDoping)"),
        ("PotentialIntrinsicCharge", "-q * IntrinsicCharge"),
    ):
        n = i[0]
        e = i[1]
        create_node_model(device, region, n, e)
        create_node_model_derivative(device, region, n, e, "Potential")

    create_quasi_fermi_levels(
        device, region, "IntrinsicElectrons", "IntrinsicHoles", variables
    )

    create_efield(device, region)
    create_dfield(device, region)

    ds.equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        node_model="PotentialIntrinsicCharge",
        edge_model="DField",
        variable_update="log_damp",
    )


def create_semiconductor_potentialOnly_contact(
    device, region, contact, is_circuit=False
):
    """
    Creates the potential equation at the semiconductor contact
    """
    # Means of determining contact charge
    # Same for all contacts

    celec_model = "(1e-10 + 0.5*abs(NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    chole_model = "(1e-10 + 0.5*abs(-NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"

    contact_model = "Potential -{} + ifelse(NetDoping > 0, \
    -V_t*log({}/NIE), \
    V_t*log({}/NIE))".format(
        get_contact_bias_name(contact), celec_model, chole_model
    )

    contact_model_name = GetContactNodeModelName(contact)
    create_contact_node_model(device, contact, contact_model_name, contact_model)

    create_contact_node_model(
        device, contact, "{}:{}".format(contact_model_name, "Potential"), "1"
    )

    ds.contact_equation(
        device=device,
        contact=contact,
        name="PotentialEquation",
        node_model=contact_model_name,
        edge_model="",
        node_charge_model="",
        edge_charge_model="DField",
        node_current_model="",
        edge_current_model="",
    )


def create_generation_recombination(
    device,
    region,
    generation_recombination_physics,
    variables,
    interface_delimiter="___",
):
    """
    Instanciates selected generation and recombination models in a consistent manner. Differentiates between bulk and surfaces. There may be multiple different surfaces.
    """

    # Initialize electron and hole recombination expressions
    Gn = "0"
    Gp = "0"

    """
    OPTICAL GENERATION
    """
    if "optical_generation" in generation_recombination_physics:
        # ds.set_parameter(device=device, name="OptScale", value=1.0)
        ds.node_solution(device=device, region=region, name="OptGen")
        xpos = ds.get_node_model_values(device=device, region=region, name="x")
        optical_generation = [0.0] * len(xpos)
        ds.set_node_values(
            device=device, region=region, name="OptGen", values=optical_generation
        )
        Gn += "+ q * OptGen"
        Gp += "- q * OptGen"

    """
    AVALANCHE GENERATION
    """
    if "avalanche" in generation_recombination_physics:
        raise NotImplementedError("Avalanche generation not implemented yet!")

    """
    BULK SHOCKLEY READ HALL
    """
    if "bulkSRH" in generation_recombination_physics:
        # Lifetime doping dependence
        taup = "taup0 / (1 + (abs(NetDoping)/Nrefp_SRH)^alpha_SRH_N)"
        taun = "taun0 / (1 + (abs(NetDoping)/Nrefn_SRH)^alpha_SRH_N)"

        # Lifetime temperature dependence
        taup += " * Tn ^ alpha_SRH_T"
        taun += " * Tn ^ alpha_SRH_T"

        # Instanciate
        USRH_bulk = f"(Electrons*Holes - NIE^2)/({taup}*(Electrons + NIE) + {taun}*(Holes + NIE))"
        Gn += " - q * USRH_bulk"
        Gp += " + q * USRH_bulk"

        # If no surface recombination, intanciate now
        if "surfaceSRH" not in generation_recombination_physics:
            create_node_model(device, region, "USRH_bulk", USRH_bulk)
            for i in ("Electrons", "Holes", "T"):
                if i in variables:
                    create_node_model_derivative(device, region, "USRH_bulk", USRH_bulk, i)

    """
    SURFACE SHOCKLEY READ HALL
    """
    if (
        "surfaceSRH" in generation_recombination_physics
    ):
        # Modify SRH to have different lifetimes on surface nodes and in bulk:
        USRH_surface = "0"
        current_interfaces = [
            interface
            for interface in ds.get_interface_list(device=device)
            if region in set(interface.split(interface_delimiter))
        ]
        for interface in current_interfaces:
            region0, region1 = interface.split(interface_delimiter)
            material0, material1 = ds.get_material(
                device=device, region=region0
            ), ds.get_material(device=device, region=region1)

            if not material0 == material1:
                at_interface_name = f"region_{region}_at_interface_{interface}"
                sp_T = f"sp_{at_interface_name} * Tn ^ eta_s_{at_interface_name}"
                sn_T = f"sn_{at_interface_name} * Tn ^ eta_s_{at_interface_name}"
                USRH_surface += f"+ (Electrons*Holes - NIE^2)/((Electrons + NIE)/{sp_T} + (Holes + NIE)/{sn_T}) * {at_interface_name}"

        Gn += " - q * USRH_surface"
        Gp += " + q * USRH_surface"

        create_node_model(device, region, "USRH_surface", USRH_surface)
        for i in ("Electrons", "Holes", "T"):
            if i in variables:
                create_node_model_derivative(
                    device, region, "USRH_surface", USRH_surface, i
                )

        # Modify bulk SRH before implementing it if present
        if "bulkSRH" in generation_recombination_physics:
            USRH_bulk += f" * region_{region}_at_bulk"
            create_node_model(device, region, "USRH_bulk", USRH_bulk)
            for i in ("Electrons", "Holes", "T"):
                if i in variables:
                    create_node_model_derivative(device, region, "USRH_bulk", USRH_bulk, i)


    """
    Create the generation-recombination
    """
    create_node_model(device, region, "ElectronGeneration", Gn)
    create_node_model(device, region, "HoleGeneration", Gp)
    for i in ("Electrons", "Holes", "T"):
        if i in variables:
            create_node_model_derivative(device, region, "ElectronGeneration", Gn, i)
            create_node_model_derivative(device, region, "HoleGeneration", Gp, i)


def create_semiconductor_electron_continuity_equation(device, region, Jn):
    """
    Electron Continuity Equation using specified equation for Jn
    """
    NCharge = "q * Electrons"
    create_node_model(device, region, "NCharge", NCharge)
    create_node_model_derivative(device, region, "NCharge", NCharge, "Electrons")

    ds.equation(
        device=device,
        region=region,
        name="ElectronContinuityEquation",
        variable_name="Electrons",
        time_node_model="NCharge",
        edge_model=Jn,
        variable_update="positive",
        node_model="ElectronGeneration",
    )


def create_semiconductor_hole_continuity_equation(device, region, Jp):
    """
    Hole Continuity Equation using specified equation for Jp
    """
    PCharge = "-q * Holes"
    create_node_model(device, region, "PCharge", PCharge)
    create_node_model_derivative(device, region, "PCharge", PCharge, "Holes")

    ds.equation(
        device=device,
        region=region,
        name="HoleContinuityEquation",
        variable_name="Holes",
        time_node_model="PCharge",
        edge_model=Jp,
        variable_update="positive",
        node_model="HoleGeneration",
    )


def create_semiconductor_potential_equation(device, region):
    """
    Create Poisson Equation assuming the Electrons and Holes as solution variables
    """
    pne = "-q*kahan3(Holes, -Electrons, NetDoping)"
    create_node_model(device, region, "PotentialNodeCharge", pne)
    create_node_model_derivative(
        device, region, "PotentialNodeCharge", pne, "Electrons"
    )
    create_node_model_derivative(device, region, "PotentialNodeCharge", pne, "Holes")

    ds.equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        node_model="PotentialNodeCharge",
        edge_model="DField",
        time_node_model="",
        variable_update="log_damp",
    )


def create_semiconductor_drift_diffusion(
    device,
    region,
    mu_n="mu_n",
    mu_p="mu_p",
    Jn="Jn",
    Jp="Jp",
    physics=None,
    interface_delimiter="___",
):
    """
    Instantiate all equations for drift diffusion simulation
    """

    physics = physics or {
        "mobility": ("doping_arora",),
        "generation_recombination": ("SRH",),
    }

    # Parse physics
    if "generation_recombination" not in physics:
        physics["generation_recombination"] = []

    # Models
    create_solution(device, region, "Potential")
    create_solution(device, region, "Electrons")
    create_solution(device, region, "Holes")
    create_efield(device, region)
    create_dfield(device, region)
    opts = create_low_field_mobility(device, region, physics["mobility"])
    opts = create_high_field_mobility(device, region, physics["mobility"], **opts)

    # Drift-diffusion per say
    create_density_of_states(device, region, physics["bandstructure"], ("Potential",))
    create_quasi_fermi_levels(
        device, region, "Electrons", "Holes", ("Electrons", "Holes", "Potential")
    )
    create_semiconductor_potential_equation(device, region)
    create_generation_recombination(
        device,
        region,
        physics["generation_recombination"],
        ("Electrons", "Holes", "Potential"),
        interface_delimiter=interface_delimiter,
    )
    create_semiconductor_electron_continuity_equation(device, region, opts["Jn"])
    create_semiconductor_hole_continuity_equation(device, region, opts["Jp"])

    # Extra models
    ds.set_node_values(
        device=device, region=region, name="Electrons", init_from="IntrinsicElectrons"
    )
    ds.set_node_values(
        device=device, region=region, name="Holes", init_from="IntrinsicHoles"
    )
    ds.element_from_edge_model(edge_model="EField", device=device, region=region)
    ds.element_model(
        device=device,
        region=region,
        name="Emag",
        equation="(EField_x^2 + EField_y^2)^(0.5)",
    )
    ds.element_from_edge_model(edge_model="Jn", device=device, region=region)
    ds.element_from_edge_model(edge_model="Jp", device=device, region=region)
    ds.element_model(
        device=device, region=region, name="Jnmag", equation="(Jn_x^2 + Jn_y^2)^(0.5)"
    )
    ds.element_model(
        device=device, region=region, name="Jpmag", equation="(Jp_x^2 + Jp_y^2)^(0.5)"
    )

    return opts


def create_semiconductor_drift_diffusion_contact(
    device, region, contact, Jn, Jp, mu_n, mu_p, is_circuit=False
):
    """
    Restrict electrons and holes to their equilibrium values
    Integrates current into circuit
    """
    create_semiconductor_potentialOnly_contact(device, region, contact, is_circuit)

    celec_model = "(1e-10 + 0.5*abs(NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    chole_model = "(1e-10 + 0.5*abs(-NetDoping+(NetDoping^2 + 4 * NIE^2)^(0.5)))"
    contact_electrons_model = (
        f"Electrons - ifelse(NetDoping > 0, {celec_model}, NIE^2/{chole_model})"
    )
    contact_holes_model = (
        f"Holes - ifelse(NetDoping < 0, +{chole_model}, +NIE^2/{celec_model})"
    )
    contact_electrons_name = f"{contact}nodeelectrons"
    contact_holes_name = f"{contact}nodeholes"

    create_contact_node_model(
        device, contact, contact_electrons_name, contact_electrons_model
    )
    create_contact_node_model(
        device, contact, "{}:{}".format(contact_electrons_name, "Electrons"), "1"
    )

    create_contact_node_model(device, contact, contact_holes_name, contact_holes_model)
    create_contact_node_model(
        device, contact, "{}:{}".format(contact_holes_name, "Holes"), "1"
    )

    if is_circuit:
        ds.contact_equation(
            device=device,
            contact=contact,
            name="ElectronContinuityEquation",
            node_model=contact_electrons_name,
            edge_current_model=Jn,
            circuit_node=get_contact_bias_name(contact),
        )

        ds.contact_equation(
            device=device,
            contact=contact,
            name="HoleContinuityEquation",
            node_model=contact_holes_name,
            edge_current_model=Jp,
            circuit_node=get_contact_bias_name(contact),
        )

    else:
        ds.contact_equation(
            device=device,
            contact=contact,
            name="ElectronContinuityEquation",
            node_model=contact_electrons_name,
            edge_current_model=Jn,
        )

        ds.contact_equation(
            device=device,
            contact=contact,
            name="HoleContinuityEquation",
            node_model=contact_holes_name,
            edge_current_model=Jp,
        )


def create_bernoulli_string(Potential="Potential", scaling_variable="V_t", sign=-1):
    """
    Creates the Bernoulli function for Scharfetter Gummel
    sign -1 for potential
    sign +1 for energy
    scaling variable should be V_t
    Potential should be scaled by V_t in V
    Ec, Ev should scaled by V_t in eV

    returns the Bernoulli expression and its argument
    Caller should understand that B(-x) = B(x) + x
    """

    if sign == -1:
        vdiff = f"({Potential}@n0 - {Potential}@n1)/{scaling_variable}"
    elif sign == 1:
        vdiff = f"({Potential}@n1 - {Potential}@n0)/{scaling_variable}"
    else:
        raise NameError(f"Invalid Sign {sign}")

    Bern01 = f"B({vdiff})"
    return (Bern01, vdiff)


def create_electron_current(
    device,
    region,
    mu_n,
    Potential="Potential",
    sign=-1,
    ElectronCurrent="ElectronCurrent",
    V_t="V_t_edge",
):
    """
    Electron current
    mu_n = mobility name
    Potential is the driving potential
    """
    ensure_edge_from_node_model_exists(device, region, "Potential")
    ensure_edge_from_node_model_exists(device, region, "Electrons")
    ensure_edge_from_node_model_exists(device, region, "Holes")
    if Potential == "Potential":
        (Bern01, vdiff) = create_bernoulli_string(
            scaling_variable=V_t, Potential=Potential, sign=sign
        )
    else:
        raise NameError("Implement proper call")

    Jn = f"q*{mu_n}*EdgeInverseLength*{V_t}*kahan3(Electrons@n1*{Bern01},  Electrons@n1*{vdiff},  -Electrons@n0*{Bern01})"

    create_edge_model(device, region, ElectronCurrent, Jn)
    for i in ("Electrons", "Potential", "Holes"):
        create_edge_model_derivatives(device, region, ElectronCurrent, Jn, i)


def create_hole_current(
    device,
    region,
    mu_p,
    Potential="Potential",
    sign=-1,
    HoleCurrent="HoleCurrent",
    V_t="V_t_edge",
):
    """
    Hole current
    """
    ensure_edge_from_node_model_exists(device, region, "Potential")
    ensure_edge_from_node_model_exists(device, region, "Electrons")
    ensure_edge_from_node_model_exists(device, region, "Holes")
    # Make sure the bernoulli functions exist
    if Potential == "Potential":
        (Bern01, vdiff) = create_bernoulli_string(
            scaling_variable=V_t, Potential=Potential, sign=sign
        )
    else:
        raise NameError("Implement proper call for " + Potential)

    Jp = f"-q*{mu_p}*EdgeInverseLength*{V_t}*kahan3(Holes@n1*{Bern01}, -Holes@n0*{Bern01}, -Holes@n0*{vdiff})"
    create_edge_model(device, region, HoleCurrent, Jp)
    for i in ("Holes", "Potential", "Electrons"):
        create_edge_model_derivatives(device, region, HoleCurrent, Jp, i)


def create_low_field_mobility(device, region, mobility_physics):
    """
    Calculate low field mobility by combining different contributions.
    """

    if "constant" in mobility_physics:
        models = (
            ("Tn", "T/300"),
            (
                "mu_n_node_lf",
                "MUN",
            ),
            (
                "mu_p_node_lf",
                "MUN",
            ),
        )

        for k, v in models:
            create_node_model(device, region, k, v)
        create_arithmetic_mean(device, region, "mu_n_node_lf", "mu_constant_n_lf")
        create_arithmetic_mean(device, region, "mu_p_node_lf", "mu_constant_p_lf")

        create_electron_current(
            device,
            region,
            mu_n="mu_constant_n_lf",
            Potential="Potential",
            sign=-1,
            ElectronCurrent="Jn_lf",
            V_t="V_t_edge",
        )
        create_hole_current(
            device,
            region,
            mu_p="mu_constant_p_lf",
            Potential="Potential",
            sign=-1,
            HoleCurrent="Jp_lf",
            V_t="V_t_edge",
        )
        return {
            "mu_n": "mu_constant_n_lf",
            "mu_p": "mu_constant_p_lf",
            "Jn": "Jn_constant_lf",
            "Jp": "Jp_constant__lf",
        }

    elif "doping_arora" in mobility_physics:
        models = (
            ("Tn", "T/300"),
            (
                "mu_n_node_lf",
                "MUMN * pow(Tn, MUMEN) + (MU0N * pow(300, -MU0EN) * pow(T, MU0EN))/(1 + pow((NTOT/(NREFN*pow(Tn, NREFNE))), ALPHA0N*pow(Tn, ALPHAEN)))",
            ),
            (
                "mu_p_node_lf",
                "MUMP * pow(Tn, MUMEP) + (MU0P * pow(300, -MU0EP) * pow(T, MU0EP))/(1 + pow((NTOT/(NREFP*pow(Tn, NREFPE))), ALPHA0P*pow(Tn, ALPHAEP)))",
            ),
        )

        for k, v in models:
            create_node_model(device, region, k, v)
        create_arithmetic_mean(device, region, "mu_n_node_lf", "mu_arora_n_lf")
        create_arithmetic_mean(device, region, "mu_p_node_lf", "mu_arora_p_lf")
        create_electron_current(
            device,
            region,
            mu_n="mu_arora_n_lf",
            Potential="Potential",
            sign=-1,
            ElectronCurrent="Jn_arora_lf",
            V_t="V_t_edge",
        )
        create_hole_current(
            device,
            region,
            mu_p="mu_arora_p_lf",
            Potential="Potential",
            sign=-1,
            HoleCurrent="Jp_arora_lf",
            V_t="V_t_edge",
        )
        return {
            "mu_n": "mu_arora_n_lf",
            "mu_p": "mu_arora_p_lf",
            "Jn": "Jn_arora_lf",
            "Jp": "Jp_arora_lf",
        }

    else:
        raise ValueError(
            'physics["mobility"] must contain one key of: constant, doping_arora'
        )


def create_high_field_mobility(device, region, mobility_physics, mu_n, mu_p, Jn, Jp):
    """
    Calculate high field mobility by modifying the low-field mobility.
    """

    if "highfield_canali" in mobility_physics:
        tlist = (
            ("vsat_n", "VSATN0 * pow(300/T, VSATNE)", ("T")),
            ("beta_n", "BETAN0 * pow(T/300, BETANE)", ("T")),
            (
                "Epar_n",
                f"ifelse(({Jn} * EField) > 0, abs(EField), 1e-15)",
                ("Potential"),
            ),
            (
                "mu_n",
                f"{mu_n} * pow(1 + pow(({mu_n}*Epar_n/vsat_n), beta_n), -1/beta_n)",
                ("Electrons", "Holes", "Potential", "T"),
            ),
            ("vsat_p", "VSATP0 * pow(300/T, VSATPE)", ("T")),
            ("beta_p", "BETAP0 * pow(T/300, BETAPE)", ("T")),
            (
                "Epar_p",
                f"ifelse(({Jp} * EField) > 0, abs(EField), 1e-15)",
                ("Potential"),
            ),
            (
                "mu_p",
                f"{mu_p} * pow(1 + pow({mu_p}*Epar_p/vsat_p, beta_p), -1/beta_p)",
                ("Electrons", "Holes", "Potential", "T"),
            ),
        )

        variable_list = ("Electrons", "Holes", "Potential")
        for model, equation, variables in tlist:
            create_edge_model(device, region, model, equation)
            for v in variable_list:
                if v in variables:
                    create_edge_model_derivatives(device, region, model, equation, v)

    else:
        tlist = (
            (
                "mu_n",
                f"{mu_n}",
                ("Electrons", "Holes", "Potential", "T"),
            ),
            (
                "mu_p",
                f"{mu_p}",
                ("Electrons", "Holes", "Potential", "T"),
            ),
        )
        variable_list = ("Electrons", "Holes", "Potential")
        for model, equation, variables in tlist:
            create_edge_model(device, region, model, equation)
            for v in variable_list:
                if v in variables:
                    create_edge_model_derivatives(device, region, model, equation, v)

    # This creates derivatives automatically
    create_electron_current(
        device,
        region,
        mu_n="mu_n",
        Potential="Potential",
        sign=-1,
        ElectronCurrent="Jn",
        V_t="V_t_edge",
    )
    create_hole_current(
        device,
        region,
        mu_p="mu_p",
        Potential="Potential",
        sign=-1,
        HoleCurrent="Jp",
        V_t="V_t_edge",
    )

    return {
        "mu_n": "mu_n",
        "mu_p": "mu_p",
        "Jn": "Jn",
        "Jp": "Jp",
    }


"""
INTERFACES
"""


def create_potentialOnly_interface(device, interface):
    """
    continuous potential at interface
    """
    model_name = create_continuous_interface_model(device, interface, "Potential")
    ds.interface_equation(
        device=device,
        interface=interface,
        name="PotentialEquation",
        interface_model=model_name,
        type="continuous",
    )


def create_semiconductor_semiconductor_interface(device, interface):
    """
    Enforces potential, electron, and hole continuity across the interface
    """
    create_potentialOnly_interface(device, interface)
    ename = create_continuous_interface_model(device, interface, "Electrons")
    ds.interface_equation(
        device=device,
        interface=interface,
        name="ElectronContinuityEquation",
        interface_model=ename,
        type="continuous",
    )
    hname = create_continuous_interface_model(device, interface, "Holes")
    ds.interface_equation(
        device=device,
        interface=interface,
        name="HoleContinuityEquation",
        interface_model=hname,
        type="continuous",
    )


def create_continuous_interface_model(device, interface, variable):
    mname = f"continuous{variable}"
    meq = "{0}@r0 - {0}@r1".format(variable)
    mname0 = f"{mname}:{variable}@r0"
    mname1 = f"{mname}:{variable}@r1"
    create_interface_model(device, interface, mname, meq)
    create_interface_model(device, interface, mname0, "1")
    create_interface_model(device, interface, mname1, "-1")
    return mname
