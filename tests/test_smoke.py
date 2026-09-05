"""
tests/test_smoke.py
====================
Fast sanity checks — not a full validation suite, but enough to catch
a broken import, a renamed field, or a NaN leaking through after any
future edit. Run with:

    python -m pytest tests/ -q
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import run_engine, INPUTS, THERMODYNAMIC_INPUTS, GEOMETRIC_DEFAULTS  # noqa: E402
from ml.correction_model import correct, FEATURES  # noqa: E402


SAMPLE_THERMO = {
    "compressor_pressure_ratio": 3.5,
    "compressor_efficiency": 0.80,
    "combustor_air_fuel_ratio": 55.0,
    "turbine_inlet_temp_K": 1050.0,
    "turbine_efficiency": 0.82,
    "mass_flow_kg_s": 0.18,
}


def test_run_engine_with_defaults_only():
    out = run_engine(**SAMPLE_THERMO)
    for f in ["thrust_N", "fuel_flow_kg_s", "tsfc_kg_N_s", "exhaust_velocity_m_s",
              "exhaust_temp_K", "compressor_power_W", "thermal_efficiency",
              "total_mass_kg", "thrust_to_weight", "lattice_surface_area_ratio"]:
        assert f in out
        assert np.isfinite(out[f]), f"{f} is not finite"
    assert out["thrust_N"] > 0
    assert out["total_mass_kg"] > 0
    assert out["thrust_to_weight"] > 0


def test_run_engine_missing_thermo_raises():
    with pytest.raises(KeyError):
        run_engine(compressor_pressure_ratio=3.0)  # missing the rest


def test_engine_uses_full_input_list_without_error():
    design = dict(GEOMETRIC_DEFAULTS)
    design.update(SAMPLE_THERMO)
    out = run_engine(**design)
    for f in INPUTS:
        assert f in out


def test_ml_correction_shapes_and_finiteness():
    out = run_engine(**SAMPLE_THERMO)
    corrected = correct(out)
    for f in FEATURES:
        assert np.isfinite(corrected[f])
        assert np.isfinite(corrected[f"physics_{f}"])
        assert np.isfinite(corrected[f"delta_{f}"])
        # corrected = physics + delta, by construction
        assert corrected[f] == pytest.approx(corrected[f"physics_{f}"] + corrected[f"delta_{f}"])


def test_cfd_dataset_columns_match_engine_inputs():
    df = pd.read_csv(PROJECT_ROOT / "data" / "cfd_dataset.csv")
    for c in INPUTS:
        assert c in df.columns, f"CFD dataset missing expected input column: {c}"


def test_optimizer_smoke():
    from optimizer.engine_optimizer import Requirements, optimize

    req = Requirements(population_size=12, generations=4, random_seed=7)
    pareto, best = optimize(req, verbose=False)
    assert len(pareto) > 0
    assert best is not None
    for f in ["compressor_pressure_ratio", "mass_flow_kg_s"]:
        assert f in best["design"]


def test_optimizer_uses_ten_thermodynamic_variables():
    from optimizer.engine_optimizer import DESIGN_VARIABLES, default_bounds

    assert len(DESIGN_VARIABLES) == 10
    bounds = default_bounds()
    for name in DESIGN_VARIABLES:
        assert name in bounds
        low, high = bounds[name]
        assert low < high


def test_optimizer_nomenclature_matches_engine_and_dataset():
    """
    The 10-variable optimizer must reuse this project's existing
    parameter names wherever an equivalent quantity already exists
    (core.engine.THERMODYNAMIC_INPUTS / data/cfd_dataset.csv columns),
    rather than inventing new names for the same physical quantities.
    """
    from core.engine import THERMODYNAMIC_INPUTS
    from optimizer.engine_optimizer import DESIGN_VARIABLES

    shared = set(DESIGN_VARIABLES) & set(THERMODYNAMIC_INPUTS)
    assert shared == {
        "compressor_pressure_ratio",
        "compressor_efficiency",
        "turbine_inlet_temp_K",
        "turbine_efficiency",
        "mass_flow_kg_s",
    }


def test_run_engine_optional_physics_knobs_match_previous_defaults():
    """
    Passing the new optional physics knobs at their default values
    must reproduce exactly the same result as not passing them at all
    (this is what keeps every existing caller of run_engine working
    unchanged).
    """
    from core.engine import PHYSICS_DEFAULTS

    baseline = run_engine(**SAMPLE_THERMO)
    with_explicit_defaults = run_engine(**SAMPLE_THERMO, **PHYSICS_DEFAULTS)
    for f in ["thrust_N", "fuel_flow_kg_s", "tsfc_kg_N_s", "exhaust_velocity_m_s",
              "exhaust_temp_K", "compressor_power_W", "thermal_efficiency"]:
        assert baseline[f] == pytest.approx(with_explicit_defaults[f])


def test_run_engine_supports_combustion_efficiency_mode():
    """
    core.engine.run_engine must support computing fuel flow from
    combustion_efficiency (air_fuel_ratio=None) — the mode the
    10-variable optimizer relies on.
    """
    design = dict(SAMPLE_THERMO)
    design.pop("combustor_air_fuel_ratio")
    design["combustor_air_fuel_ratio"] = None
    design["combustion_efficiency"] = 0.98
    out = run_engine(**design)
    assert np.isfinite(out["fuel_flow_kg_s"])
    assert out["fuel_flow_kg_s"] > 0
