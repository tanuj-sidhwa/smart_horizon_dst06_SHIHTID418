"""
core.geometry
=============
Preliminary sizing / mass-estimation model.

WHY THIS FILE EXISTS
---------------------
The original thermodynamic cycle model (inlet -> compressor -> combustor
-> turbine -> nozzle -> performance) only ever consumed 6 of the ~19
design-vector fields (pressure ratio, compressor efficiency, mass flow,
turbine inlet temperature, turbine efficiency, air-fuel ratio). Every
geometric field (diameters, blade counts, combustor dimensions,
injector count, hub-to-tip ratio, nozzle exit diameter, rpm, lattice
params) was accepted as an input but never used — `total_mass_kg`,
`thrust_to_weight` and `lattice_surface_area_ratio` were hard-coded to
0.0 in the old engine.py.

That's fine for a pure thermodynamic performance study, but it breaks
two things you need:
  1. A real thrust-to-weight objective for the optimizer.
  2. Self-consistent geometry to hand to the CAD generator — otherwise
     the optimizer could return a design where the "optimal" geometry
     has no bearing on the optimal thermodynamics.

This module closes that gap with closed-form, fast, engineering-level
approximations (thin-shell mass estimates per section, from the
geometric design vector). They are deliberately simple so they can run
thousands of times inside the NSGA-II loop. They are NOT a substitute
for a structural/FEA sizing pass — treat their output as a scaffolding
estimate for CAD and for the thrust/weight ranking, and refine later.
"""

import math

from .constants import (
    DENSITY_COMPRESSOR_ALLOY_KG_M3,
    DENSITY_COMBUSTOR_ALLOY_KG_M3,
    DENSITY_TURBINE_ALLOY_KG_M3,
    DENSITY_NOZZLE_ALLOY_KG_M3,
    COMPRESSOR_SOLIDITY_FRACTION,
    COMBUSTOR_SOLIDITY_FRACTION,
    TURBINE_SOLIDITY_FRACTION,
    NOZZLE_SOLIDITY_FRACTION,
    ANCILLARY_MASS_FRACTION,
    LATTICE_SA_V_COEFFICIENT,
    STANDARD_GRAVITY,
)


def _cylindrical_shell_mass(diameter_mm, length_mm, wall_thickness_mm, density_kg_m3):
    """Mass of a thin hollow cylinder: 2*pi*r*t*L*rho (r = mean radius)."""
    d_m = diameter_mm / 1000.0
    l_m = length_mm / 1000.0
    t_m = wall_thickness_mm / 1000.0
    mean_radius_m = max(d_m / 2.0 - t_m / 2.0, 0.0)
    volume_m3 = 2.0 * math.pi * mean_radius_m * t_m * l_m
    return volume_m3 * density_kg_m3


def compressor_mass_kg(compressor_diameter_mm, compressor_blade_count):
    """
    Rotor + casing mass estimate. Length is approximated from diameter
    (typical axial/centrifugal small-turbojet compressor stage aspect
    ratio ~0.35 of diameter), wall/disc thickness scales mildly with
    blade count (more blades -> slightly thicker hub for root fixing).
    """
    length_mm = 0.35 * compressor_diameter_mm
    wall_mm = max(1.5, 0.03 * compressor_diameter_mm + 0.05 * compressor_blade_count)
    shell = _cylindrical_shell_mass(
        compressor_diameter_mm, length_mm, wall_mm, DENSITY_COMPRESSOR_ALLOY_KG_M3
    )
    # Blades + hub add mass beyond the bare shell estimate (see module docstring).
    return shell * (1.0 + COMPRESSOR_SOLIDITY_FRACTION)


def combustor_mass_kg(
    combustor_length_mm,
    combustor_outer_diameter_mm,
    combustor_inner_diameter_mm,
    combustor_liner_thickness_mm,
    combustor_num_injectors,
):
    """Outer casing + inner liner + injector hardware."""
    outer_wall_mm = max(1.0, 0.02 * combustor_outer_diameter_mm)
    casing = _cylindrical_shell_mass(
        combustor_outer_diameter_mm,
        combustor_length_mm,
        outer_wall_mm,
        DENSITY_COMBUSTOR_ALLOY_KG_M3,
    )
    liner = _cylindrical_shell_mass(
        combustor_inner_diameter_mm,
        combustor_length_mm,
        combustor_liner_thickness_mm,
        DENSITY_COMBUSTOR_ALLOY_KG_M3,
    )
    injector_mass_kg = combustor_num_injectors * 0.015  # ~15 g per injector, typical small swirl injector
    return (casing + liner) * (1.0 + COMBUSTOR_SOLIDITY_FRACTION) + injector_mass_kg


def turbine_mass_kg(nozzle_exit_diameter_mm, turbine_blade_count, turbine_hub_tip_ratio):
    """
    Turbine disc/blade group mass. Diameter isn't a direct design field
    for the turbine, so it's approximated from the nozzle exit diameter
    (nozzle throat is downstream of, and typically close in scale to,
    the turbine annulus for this engine class) inflated by the inverse
    hub-to-tip ratio (a lower hub:tip ratio means taller/longer blades
    for the same hub, i.e. a larger effective disc diameter).
    """
    tip_diameter_mm = nozzle_exit_diameter_mm / max(turbine_hub_tip_ratio, 0.3)
    length_mm = 0.25 * tip_diameter_mm
    wall_mm = max(1.5, 0.04 * tip_diameter_mm + 0.03 * turbine_blade_count)
    shell = _cylindrical_shell_mass(
        tip_diameter_mm, length_mm, wall_mm, DENSITY_TURBINE_ALLOY_KG_M3
    )
    return shell * (1.0 + TURBINE_SOLIDITY_FRACTION)


def nozzle_mass_kg(nozzle_exit_diameter_mm):
    """Convergent exhaust cone, length approximated from exit diameter."""
    length_mm = 1.2 * nozzle_exit_diameter_mm
    wall_mm = max(0.8, 0.015 * nozzle_exit_diameter_mm)
    shell = _cylindrical_shell_mass(
        nozzle_exit_diameter_mm, length_mm, wall_mm, DENSITY_NOZZLE_ALLOY_KG_M3
    )
    return shell * (1.0 + NOZZLE_SOLIDITY_FRACTION)


def lattice_surface_area_ratio(lattice_cell_size_mm, lattice_density):
    """
    Heuristic surface-area-to-volume ratio [1/mm] contributed by a
    cubic lattice infill of the given unit-cell size and relative
    density (0-1). Used only for lightweight/printed bracket
    components, not pressure boundaries. Higher ratio => more surface
    for heat rejection / less bulk mass per unit stiffness.
    """
    cell = max(lattice_cell_size_mm, 1e-6)
    return LATTICE_SA_V_COEFFICIENT * lattice_density / cell


def estimate_geometry_and_mass(p):
    """
    p: dict containing (at least) the geometric design-vector fields:
        compressor_diameter_mm, compressor_blade_count,
        combustor_length_mm, combustor_outer_diameter_mm,
        combustor_inner_diameter_mm, combustor_liner_thickness_mm,
        combustor_num_injectors, nozzle_exit_diameter_mm,
        turbine_blade_count, turbine_hub_tip_ratio,
        lattice_cell_size_mm, lattice_density

    Returns a dict with:
        compressor_mass_kg, combustor_mass_kg, turbine_mass_kg,
        nozzle_mass_kg, ancillary_mass_kg, total_mass_kg,
        lattice_surface_area_ratio,
        overall_length_mm, overall_max_diameter_mm
    """
    m_comp = compressor_mass_kg(p["compressor_diameter_mm"], p["compressor_blade_count"])
    m_comb = combustor_mass_kg(
        p["combustor_length_mm"],
        p["combustor_outer_diameter_mm"],
        p["combustor_inner_diameter_mm"],
        p["combustor_liner_thickness_mm"],
        p["combustor_num_injectors"],
    )
    m_turb = turbine_mass_kg(
        p["nozzle_exit_diameter_mm"], p["turbine_blade_count"], p["turbine_hub_tip_ratio"]
    )
    m_noz = nozzle_mass_kg(p["nozzle_exit_diameter_mm"])

    subtotal = m_comp + m_comb + m_turb + m_noz
    m_ancillary = subtotal * ANCILLARY_MASS_FRACTION
    total_mass_kg = subtotal + m_ancillary

    lattice_ratio = lattice_surface_area_ratio(
        p["lattice_cell_size_mm"], p["lattice_density"]
    )

    inlet_length_mm = 0.3 * p["compressor_diameter_mm"]
    turbine_length_mm = 0.25 * (p["nozzle_exit_diameter_mm"] / max(p["turbine_hub_tip_ratio"], 0.3))
    nozzle_length_mm = 1.2 * p["nozzle_exit_diameter_mm"]
    overall_length_mm = (
        inlet_length_mm
        + 0.35 * p["compressor_diameter_mm"]
        + p["combustor_length_mm"]
        + turbine_length_mm
        + nozzle_length_mm
    )
    overall_max_diameter_mm = max(
        p["compressor_diameter_mm"],
        p["combustor_outer_diameter_mm"],
        p["nozzle_exit_diameter_mm"] / max(p["turbine_hub_tip_ratio"], 0.3),
    )

    return {
        "compressor_mass_kg": m_comp,
        "combustor_mass_kg": m_comb,
        "turbine_mass_kg": m_turb,
        "nozzle_mass_kg": m_noz,
        "ancillary_mass_kg": m_ancillary,
        "total_mass_kg": total_mass_kg,
        "lattice_surface_area_ratio": lattice_ratio,
        "overall_length_mm": overall_length_mm,
        "overall_max_diameter_mm": overall_max_diameter_mm,
    }


def thrust_to_weight(thrust_N, total_mass_kg):
    if total_mass_kg <= 0:
        return 0.0
    weight_N = total_mass_kg * STANDARD_GRAVITY
    return thrust_N / weight_N
