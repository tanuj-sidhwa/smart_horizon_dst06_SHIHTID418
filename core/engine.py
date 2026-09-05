"""
core.engine
===========
Top-level orchestrator: takes ONE full design vector (a dict of the
fields in `INPUTS`) and runs the entire 0D thermodynamic cycle plus
the preliminary geometry/mass sizing pass, returning a single flat
dict containing every input echoed back alongside every predicted
performance and geometric output.

This is the ONE function every other part of the project calls:
  - pipeline/validate_against_cfd.py  (CFD-input replay + physics)
  - optimizer/nsga2.py                (candidate evaluation)
  - app/streamlit_app.py              (interactive UI)
  - cad/freecad_generate.py           (reads the dict this produces)

Design vector fields
---------------------
THERMODYNAMIC (these actually change the cycle solution):
    compressor_pressure_ratio, compressor_efficiency,
    combustor_air_fuel_ratio, turbine_inlet_temp_K, turbine_efficiency,
    mass_flow_kg_s

GEOMETRIC (these do not change the cycle solution, but drive the
preliminary sizing pass in core.geometry and are required by the CAD
generator):
    compressor_diameter_mm, compressor_blade_count,
    combustor_length_mm, combustor_outer_diameter_mm,
    combustor_inner_diameter_mm, combustor_liner_thickness_mm,
    combustor_num_injectors, turbine_blade_count, turbine_hub_tip_ratio,
    nozzle_exit_diameter_mm, rpm, lattice_cell_size_mm, lattice_density

Both groups are listed in INPUTS below (order matches the CFD dataset
columns) so the same design vector can be round-tripped between the
CFD CSV, the physics model, the ML corrector, the optimizer, and the
CAD generator without any renaming.

EXTRA (non-CFD-dataset-backed) PHYSICS KNOBS
---------------------------------------------
Five additional cycle knobs are supported as *optional* keyword
arguments, each with a default equal to the value this module used to
hard-code internally: `inlet_pressure_recovery`,
`combustor_pressure_drop_fraction`, `combustion_efficiency`,
`nozzle_discharge_coefficient`, `nozzle_isentropic_efficiency`. They
are intentionally NOT part of `INPUTS`/`THERMODYNAMIC_INPUTS` because
`data/cfd_dataset.csv` (and therefore the trained ML corrector's
training domain) does not vary them — but optimizer/engine_optimizer.py's
10-variable KJ66-referenced search exposes them as real design
variables. Any caller that does not pass them (existing tests, the
CFD-replay pipeline, the CAD flow) gets bit-for-bit the same behaviour
as before this was added.
"""

from .inlet import inlet_model
from .compressor import compressor_model
from .combustor import combustor_model
from .turbine import turbine_model
from .nozzle import nozzle_model
from .performance import performance_model
from .geometry import estimate_geometry_and_mass, thrust_to_weight

# Column order matches data/cfd_dataset.csv input columns exactly.
INPUTS = [
    "compressor_pressure_ratio",
    "compressor_efficiency",
    "compressor_diameter_mm",
    "compressor_blade_count",
    "combustor_length_mm",
    "combustor_outer_diameter_mm",
    "combustor_inner_diameter_mm",
    "combustor_liner_thickness_mm",
    "combustor_num_injectors",
    "combustor_air_fuel_ratio",
    "turbine_inlet_temp_K",
    "turbine_efficiency",
    "turbine_blade_count",
    "turbine_hub_tip_ratio",
    "nozzle_exit_diameter_mm",
    "mass_flow_kg_s",
    "rpm",
    "lattice_cell_size_mm",
    "lattice_density",
]

# The subset of INPUTS that actually drives the thermodynamic cycle.
THERMODYNAMIC_INPUTS = [
    "compressor_pressure_ratio",
    "compressor_efficiency",
    "combustor_air_fuel_ratio",
    "turbine_inlet_temp_K",
    "turbine_efficiency",
    "mass_flow_kg_s",
]

# The subset of INPUTS used only by the geometry/mass sizing pass.
GEOMETRIC_INPUTS = [f for f in INPUTS if f not in THERMODYNAMIC_INPUTS]

# Reasonable defaults for geometric fields, used when a caller (e.g.
# the optimizer, which only searches THERMODYNAMIC_INPUTS) doesn't
# supply them. Loosely sized around a KJ-66-class small turbojet.
GEOMETRIC_DEFAULTS = {
    "compressor_diameter_mm": 66.0,
    "compressor_blade_count": 12,
    "combustor_length_mm": 90.0,
    "combustor_outer_diameter_mm": 100.0,
    "combustor_inner_diameter_mm": 60.0,
    "combustor_liner_thickness_mm": 1.2,
    "combustor_num_injectors": 4,
    "turbine_blade_count": 19,
    "turbine_hub_tip_ratio": 0.65,
    "nozzle_exit_diameter_mm": 50.0,
    "rpm": 108000.0,
    "lattice_cell_size_mm": 3.0,
    "lattice_density": 0.35,
}

# Optional thermodynamic-cycle knobs NOT present in data/cfd_dataset.csv.
# Defaults equal the values this module used to hard-code inline, so
# omitting them reproduces the exact previous behaviour. Pass any of
# these explicitly (e.g. from optimizer/engine_optimizer.py's 10-variable
# KJ66-referenced search) to actually vary them.
PHYSICS_DEFAULTS = {
    "inlet_pressure_recovery": 0.98,
    "combustor_pressure_drop_fraction": 0.05,
    "combustion_efficiency": 0.98,
    "nozzle_discharge_coefficient": 0.98,
    "nozzle_isentropic_efficiency": 0.95,
}


def run_engine(**p):
    """
    Run the full engine model for one design point.

    Accepts every field in THERMODYNAMIC_INPUTS as required keyword
    arguments; any field in GEOMETRIC_INPUTS that is missing falls
    back to GEOMETRIC_DEFAULTS so this function also works when called
    with only the 6 thermodynamic variables (e.g. from the optimizer).

    Returns a flat dict: every input echoed back + every predicted
    output (thrust, TSFC, thermal efficiency, mass, thrust-to-weight,
    geometry, etc.).
    """
    design = dict(GEOMETRIC_DEFAULTS)
    design.update(PHYSICS_DEFAULTS)
    design.update(p)

    missing = [f for f in THERMODYNAMIC_INPUTS if f not in design]
    if missing:
        raise KeyError(f"run_engine missing required thermodynamic input(s): {missing}")

    # ---- 1. Thermodynamic cycle -----------------------------------
    inlet = inlet_model(inlet_pressure_recovery=design["inlet_pressure_recovery"])
    comp = compressor_model(
        inlet["Tt2"], inlet["Pt2"],
        design["compressor_pressure_ratio"], design["compressor_efficiency"],
    )
    comb = combustor_model(
        design["mass_flow_kg_s"], comp["Tt3"], comp["Pt3"],
        design["turbine_inlet_temp_K"],
        design["combustor_pressure_drop_fraction"],
        design["combustion_efficiency"],
        design.get("combustor_air_fuel_ratio"),
    )
    turb = turbine_model(
        comb["Tt4"], comb["Pt4"], comp["specific_work"], design["turbine_efficiency"],
    )
    noz = nozzle_model(
        turb["Tt5"], turb["Pt5"],
        discharge_coefficient=design["nozzle_discharge_coefficient"],
        isentropic_efficiency=design["nozzle_isentropic_efficiency"],
    )

    compressor_power_W = comp["specific_work"] * design["mass_flow_kg_s"]
    perf = performance_model(
        design["mass_flow_kg_s"], comb["fuel_flow"], noz["V9"], noz["T9"], compressor_power_W,
    )

    # ---- 2. Preliminary geometry / mass sizing ---------------------
    geom = estimate_geometry_and_mass(design)

    # ---- 3. Assemble flat output dict ------------------------------
    out = dict(design)
    out.update(perf)
    out["specific_thrust"] = out["thrust_N"] / design["mass_flow_kg_s"]
    out.update(geom)
    out["thrust_to_weight"] = thrust_to_weight(out["thrust_N"], geom["total_mass_kg"])

    return out
