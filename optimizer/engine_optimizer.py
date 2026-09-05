"""
optimizer.engine_optimizer
===========================
Requirements -> design translator, updated to the KJ66-referenced
10-variable thermodynamic search (see nsga2_kj66_integrated.py, which
this module folds into the project's existing architecture and
nomenclature).

You give it what you WANT the engine to do (target thrust, an
allowable thrust band, a max exhaust temperature, an allowable exhaust
velocity band) and it runs NSGA-II (optimizer.nsga2.NSGA2) over the
10 thermodynamic-cycle design variables below through:

    design -> core.engine.run_engine (physics)
           -> ml.correction_model.correct (CFD-corrected performance)
           -> objectives / constraints (from your Requirements)
           -> NSGA-II

...and returns the Pareto-optimal set of designs plus one recommended
"best compromise" design, ready to export for CAD (cad/freecad_generate.py).

10 DESIGN VARIABLES (same names as core.engine / data/cfd_dataset.csv
wherever an equivalent quantity already existed there, so nomenclature
stays consistent project-wide):
    1. inlet_pressure_recovery            (new — core.inlet knob)
    2. compressor_pressure_ratio          (CFD-dataset-backed)
    3. compressor_efficiency              (CFD-dataset-backed)
    4. turbine_inlet_temp_K               (CFD-dataset-backed; = combustor
                                            exit temperature)
    5. combustor_pressure_drop_fraction   (new — core.combustor knob)
    6. combustion_efficiency              (new — core.combustor knob)
    7. turbine_efficiency                 (CFD-dataset-backed)
    8. nozzle_discharge_coefficient       (new — core.nozzle knob)
    9. nozzle_isentropic_efficiency       (new — core.nozzle knob)
   10. mass_flow_kg_s                     (CFD-dataset-backed)

    NOTE ON RENAMING: the reference NSGA-II script this was ported from
    used its own local names for four of these
    (compressor_isentropic_efficiency, combustor_exit_temperature_K,
    turbine_isentropic_efficiency, mass_flow_air_kg_s). Those are
    renamed here to compressor_efficiency, turbine_inlet_temp_K,
    turbine_efficiency, and mass_flow_kg_s respectively, because those
    are the names already used everywhere else in this project
    (core/engine.py's THERMODYNAMIC_INPUTS, data/cfd_dataset.csv
    columns, ml/model_config.json provenance, tests/test_smoke.py,
    app/streamlit_app.py). No other project file needs to change
    because of this rename — the physics is identical either way.

Objectives (all minimized internally; efficiency is negated):
    1. |thrust - target_thrust|
    2. TSFC
    3. -thermal_efficiency
    4. fuel_flow
    5. compressor_power   <-- CHANGED from -thrust_to_weight (see below)

Constraints (satisfied when <= 0):
    1. exhaust_temp - max_exhaust_temp
    2. min_exhaust_velocity - exhaust_velocity
    3. exhaust_velocity - max_exhaust_velocity
    4. min_thrust - thrust
    5. thrust - max_thrust

WHAT ACTUALLY CHANGED vs. the previous version of this file
-------------------------------------------------------------
1. Search space: 10 pure thermodynamic-cycle variables instead of the
   old 6 thermodynamic + 13 geometric = 19 variables. Geometry is no
   longer searched; it is held at core.engine.GEOMETRIC_DEFAULTS (a
   KJ-66-class scaffold) so CAD export and mass/thrust-to-weight
   reporting still work unchanged — they are just no longer being
   optimized.
2. Objective 5 is now `compressor_power` (minimize), not
   `-thrust_to_weight` (maximize), because thrust-to-weight depends on
   geometry, which is no longer part of the search. Thrust-to-weight
   is still computed and reported (using the default geometry) for
   information, but it is not something the optimizer pushes on
   anymore.
3. `combustor_air_fuel_ratio` is no longer a search variable. Fuel
   flow is instead computed from `combustion_efficiency` +
   `turbine_inlet_temp_K` (core.combustor.combustor_model's other
   supported mode). core.engine.run_engine now accepts
   `combustor_air_fuel_ratio=None` to select this mode explicitly.
4. Five new cycle knobs are exposed that the old optimizer held fixed
   internally: `inlet_pressure_recovery`, `combustor_pressure_drop_fraction`,
   `combustion_efficiency`, `nozzle_discharge_coefficient`,
   `nozzle_isentropic_efficiency`. core/engine.py's PHYSICS_DEFAULTS
   gives them the exact same values they used to be hard-coded to, so
   anything that does NOT search them (this optimizer's old callers,
   CFD replay, tests) behaves exactly as before.
5. Bounds: the 5 dataset-backed variables now use the *actual*
   data/cfd_dataset.csv min/max span (built at run time, see
   `default_bounds()`) instead of a hand-picked static range, matching
   the reference script's approach. The 5 new variables use
   engineering ranges around the KJ66 reference baseline
   (KJ66_BASELINE below), since they are not in the CFD dataset.
6. The KJ66 reference baseline is now seeded into generation 0
   (optimizer.nsga2.NSGA2's `seed_designs`) instead of the initial
   population being 100% random.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import run_engine, GEOMETRIC_DEFAULTS  # noqa: E402
from ml.correction_model import correct  # noqa: E402
from optimizer.nsga2 import NSGA2, Individual  # noqa: E402

DATA_FILE = PROJECT_ROOT / "data" / "cfd_dataset.csv"

# --------------------------------------------------------------------
# The exact ten design variables searched by this optimizer.
# --------------------------------------------------------------------
DESIGN_VARIABLES: Tuple[str, ...] = (
    "inlet_pressure_recovery",
    "compressor_pressure_ratio",
    "compressor_efficiency",
    "turbine_inlet_temp_K",
    "combustor_pressure_drop_fraction",
    "combustion_efficiency",
    "turbine_efficiency",
    "nozzle_discharge_coefficient",
    "nozzle_isentropic_efficiency",
    "mass_flow_kg_s",
)

# No integer-valued fields in this 10-variable thermodynamic search
# (unlike the old 19-variable search, which had blade/injector counts).
INT_FIELDS: Tuple[str, ...] = ()

# CFD-dataset columns backing 5 of the 10 design variables. Because
# this project's nomenclature is already shared between core.engine
# and data/cfd_dataset.csv, the design-variable name equals the
# dataset column name for every one of these.
_DATASET_BACKED_VARIABLES = (
    "compressor_pressure_ratio",
    "compressor_efficiency",
    "turbine_inlet_temp_K",
    "turbine_efficiency",
    "mass_flow_kg_s",
)

# Engineering ranges for the 5 variables that are not represented in
# data/cfd_dataset.csv (core.engine.PHYSICS_DEFAULTS gives their
# fixed/default values; these are the *search* ranges around them).
_ENGINEERING_RANGE_VARIABLES: Dict[str, Tuple[float, float]] = {
    "inlet_pressure_recovery": (0.95, 0.99),
    "combustor_pressure_drop_fraction": (0.03, 0.08),
    "combustion_efficiency": (0.90, 0.99),
    "nozzle_discharge_coefficient": (0.95, 1.00),
    "nozzle_isentropic_efficiency": (0.90, 0.98),
}

# KJ66 / engineering reference baseline, used to seed generation 0 and
# to sanity-check that the search bounds actually contain it.
# Public KJ66 reference values available in the literature:
#   compressor pressure ratio ~= 2.2, design mass flow ~= 0.22 kg/s,
#   design thrust ~= 75 N. The remaining variables are not published
#   as one authoritative KJ66 table, so engineering/reference values
#   already used elsewhere in this project are used instead.
KJ66_BASELINE: Dict[str, float] = {
    "inlet_pressure_recovery": 0.98,
    "compressor_pressure_ratio": 2.20,
    "compressor_efficiency": 0.73,
    "turbine_inlet_temp_K": 1000.0,
    "combustor_pressure_drop_fraction": 0.05,
    "combustion_efficiency": 0.98,
    "turbine_efficiency": 0.85,
    "nozzle_discharge_coefficient": 0.98,
    "nozzle_isentropic_efficiency": 0.95,
    "mass_flow_kg_s": 0.22,
}


def _data_range(df: pd.DataFrame, column: str) -> Optional[Tuple[float, float]]:
    if column not in df.columns:
        return None
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return None
    low, high = float(values.min()), float(values.max())
    if not np.isfinite([low, high]).all() or low >= high:
        return None
    return low, high


@lru_cache(maxsize=1)
def default_bounds() -> Dict[str, Tuple[float, float]]:
    """
    Build the 10-variable design space.

    The 5 CFD-dataset-backed variables use the actual span of
    data/cfd_dataset.csv. The remaining 5 use the engineering ranges
    in `_ENGINEERING_RANGE_VARIABLES`. Cached because the dataset scan
    only needs to happen once per process.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"CFD dataset not found:\n{DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    if "is_valid" in df.columns:
        valid = df["is_valid"].astype(str).str.upper().isin(["TRUE", "1", "YES"])
        if valid.any():
            df = df.loc[valid].copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    bounds: Dict[str, Tuple[float, float]] = {}
    for name in _DATASET_BACKED_VARIABLES:
        rng = _data_range(df, name)
        if rng is None:
            raise ValueError(f"Required CFD column '{name}' is missing or invalid.")
        bounds[name] = rng

    bounds.update(_ENGINEERING_RANGE_VARIABLES)

    for name in DESIGN_VARIABLES:
        if name not in bounds:
            raise ValueError(f"No bound defined for {name}.")
        low, high = bounds[name]
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Invalid bound for {name}: ({low}, {high})")

    return bounds


def verify_baseline_in_bounds(bounds: Dict[str, Tuple[float, float]]) -> List[str]:
    """Returns a list of human-readable warnings for baseline values outside bounds."""
    outside = []
    for name in DESIGN_VARIABLES:
        if name not in bounds:
            continue
        value = KJ66_BASELINE[name]
        low, high = bounds[name]
        if value < low or value > high:
            outside.append(f"{name}: baseline={value}, bounds=({low}, {high})")
    return outside


@dataclass
class Requirements:
    target_thrust_N: float = 100.0
    thrust_band_N: float = 3.0            # +/- band around target -> constraint
    max_exhaust_temp_K: float = 1200.0
    min_exhaust_velocity_m_s: float = 250.0
    max_exhaust_velocity_m_s: float = 450.0

    population_size: int = 80
    generations: int = 50
    random_seed: int = 42

    # Optional: override the auto-built (CFD-dataset + engineering
    # range) bounds, or hold some of the 10 fields fixed at given
    # values (they are removed from the search and taken from
    # `fixed_fields` on every evaluation instead).
    bounds_override: Optional[Dict[str, Tuple[float, float]]] = None
    fixed_fields: Dict[str, float] = field(default_factory=dict)

    @property
    def min_thrust_N(self) -> float:
        return self.target_thrust_N - self.thrust_band_N

    @property
    def max_thrust_N(self) -> float:
        return self.target_thrust_N + self.thrust_band_N


def _evaluate(design: Dict[str, float], req: Requirements, fixed: Dict[str, float]):
    full_design = dict(fixed)
    full_design.update(design)
    # The 10-variable search does not use an explicit air/fuel ratio;
    # core.engine.run_engine falls back to computing fuel flow from
    # combustion_efficiency + turbine_inlet_temp_K when this is None.
    full_design.setdefault("combustor_air_fuel_ratio", None)

    physics = run_engine(**full_design)
    corrected = correct(physics)

    thrust = corrected["thrust_N"]
    fuel_flow = corrected["fuel_flow_kg_s"]
    tsfc = corrected["tsfc_kg_N_s"]
    exhaust_temp = corrected["exhaust_temp_K"]
    exhaust_velocity = corrected["exhaust_velocity_m_s"]
    thermal_efficiency = corrected["thermal_efficiency"]
    compressor_power = corrected["compressor_power_W"]

    objectives = (
        abs(thrust - req.target_thrust_N),
        tsfc,
        -thermal_efficiency,
        fuel_flow,
        compressor_power,
    )

    constraints = (
        exhaust_temp - req.max_exhaust_temp_K,
        req.min_exhaust_velocity_m_s - exhaust_velocity,
        exhaust_velocity - req.max_exhaust_velocity_m_s,
        req.min_thrust_N - thrust,
        thrust - req.max_thrust_N,
    )

    extra = {"physics": physics, "corrected": corrected}
    return objectives, constraints, extra


def optimize(req: Requirements, verbose: bool = True) -> Tuple[List[Individual], Dict]:
    full_bounds = dict(req.bounds_override or default_bounds())

    if verbose:
        for warning in verify_baseline_in_bounds(full_bounds):
            print(f"  WARNING: KJ66 baseline outside bounds -> {warning}")

    bounds = dict(full_bounds)
    for f in req.fixed_fields:
        bounds.pop(f, None)

    seed_designs = []
    baseline = {k: v for k, v in KJ66_BASELINE.items() if k in bounds}
    if baseline:
        seed_designs.append(baseline)

    nsga = NSGA2(
        bounds=bounds,
        evaluate_fn=lambda d: _evaluate(d, req, req.fixed_fields),
        population_size=req.population_size,
        generations=req.generations,
        mutation_prob=0.15,
        random_seed=req.random_seed,
        int_fields=INT_FIELDS,
        seed_designs=seed_designs,
    )

    if verbose:
        print(f"Running NSGA-II: pop={req.population_size}, gens={req.generations}, "
              f"variables={len(bounds)}")

    final_pop = nsga.run(verbose=verbose)
    pareto = NSGA2.pareto_front(final_pop)

    best = pick_best_compromise(pareto, req)
    return pareto, best


def pick_best_compromise(pareto: List[Individual], req: Requirements) -> Optional[Dict]:
    """
    Among feasible Pareto-optimal individuals, pick the one closest to
    the thrust target, breaking ties by lowest TSFC. If none are
    feasible, fall back to the least-infeasible individual.
    """
    if not pareto:
        return None

    feasible = [p for p in pareto if p.feasible]
    pool = feasible if feasible else pareto

    def score(ind: Individual):
        thrust_error = ind.objectives[0]
        tsfc = ind.objectives[1]
        violation = ind.violation
        return (violation, thrust_error, tsfc)

    best = min(pool, key=score)
    # Geometry is not searched by this 10-variable optimizer, so it is
    # not in `best.design`. Include core.engine.GEOMETRIC_DEFAULTS
    # explicitly in the exported "design" so best_design.json still
    # carries every field cad/freecad_generate.py needs.
    full_design = dict(GEOMETRIC_DEFAULTS)
    full_design.update(req.fixed_fields)
    full_design.update(best.design)
    full_design.setdefault("combustor_air_fuel_ratio", None)
    physics = run_engine(**full_design)
    corrected = correct(physics)

    return {
        "design": full_design,
        "physics": physics,
        "corrected": corrected,
        "objectives": best.objectives,
        "constraints": best.constraints,
        "feasible": best.feasible,
    }


def pareto_to_dataframe(pareto: List[Individual], req: Requirements) -> pd.DataFrame:
    rows = []
    for ind in pareto:
        full_design = dict(GEOMETRIC_DEFAULTS)
        full_design.update(req.fixed_fields)
        full_design.update(ind.design)
        full_design.setdefault("combustor_air_fuel_ratio", None)
        physics = ind.extra.get("physics") or run_engine(**full_design)
        corrected = ind.extra.get("corrected") or correct(physics)

        row = dict(full_design)
        row["feasible"] = ind.feasible
        row["thrust_error_N"] = ind.objectives[0]
        row["tsfc_kg_N_s"] = corrected["tsfc_kg_N_s"]
        row["thermal_efficiency"] = corrected["thermal_efficiency"]
        row["fuel_flow_kg_s"] = corrected["fuel_flow_kg_s"]
        row["compressor_power_W"] = corrected["compressor_power_W"]
        row["thrust_to_weight"] = physics["thrust_to_weight"]
        row["thrust_N_corrected"] = corrected["thrust_N"]
        row["exhaust_temp_K_corrected"] = corrected["exhaust_temp_K"]
        row["exhaust_velocity_m_s_corrected"] = corrected["exhaust_velocity_m_s"]
        row["total_mass_kg"] = physics["total_mass_kg"]
        rows.append(row)
    return pd.DataFrame(rows)


# Backwards-compatible name: earlier versions of this module exposed a
# module-level DEFAULT_BOUNDS dict. Anything importing that name still
# works; it now resolves the CFD-dataset-backed bounds at import time.
DEFAULT_BOUNDS = default_bounds()
