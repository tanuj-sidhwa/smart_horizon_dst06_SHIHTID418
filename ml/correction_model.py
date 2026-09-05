"""
ml.correction_model
====================
Loads the trained physics->CFD correction model (HistGradientBoosting,
selected by validation RMSE in the original training run over Ridge /
RandomForest / ExtraTrees / HistGradientBoosting — see model_config.json)
and exposes one function: `correct(physics_output_dict) -> corrected_dict`.

Workflow (must match train_physics_correction.py exactly):
    physics performance P (7 features)
          -> input_scaler.transform
          -> correction_model.predict   (scaled space)
          -> target_scaler.inverse_transform  =>  Delta (CFD - Physics)
          -> corrected = P + Delta

The 7 corrected features are: thrust_N, fuel_flow_kg_s, tsfc_kg_N_s,
exhaust_velocity_m_s, exhaust_temp_K, compressor_power_W,
thermal_efficiency.

Everything else in the engine's output dict (geometry, mass,
thrust-to-weight, all design inputs) passes through unchanged — the ML
model was only ever trained to correct those 7 thermodynamic
performance numbers against CFD.
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

# The bundled model artifacts were pickled with a slightly different
# scikit-learn version than may be installed here. This is a version
# *warning*, not a correctness issue for these estimators — but if you
# see numerically odd results after a scikit-learn upgrade, retrain
# with ml/train_physics_correction.py against the current environment.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

_HERE = Path(__file__).resolve().parent

MODEL_FILE = _HERE / "correction_model.joblib"
INPUT_SCALER_FILE = _HERE / "input_scaler.joblib"
TARGET_SCALER_FILE = _HERE / "target_scaler.joblib"
CONFIG_FILE = _HERE / "model_config.json"

for f in (MODEL_FILE, INPUT_SCALER_FILE, TARGET_SCALER_FILE):
    if not f.exists():
        raise FileNotFoundError(
            f"Missing ML correction artifact: {f}\n"
            "Run ml/train_physics_correction.py first (or copy the "
            "artifacts from the original ML_TRIAL run) to populate this "
            "directory."
        )

with open(CONFIG_FILE, "r", encoding="utf-8") as fh:
    MODEL_CONFIG = json.load(fh)

FEATURES = MODEL_CONFIG["features"]  # order matters — must match training

_model = joblib.load(MODEL_FILE)
_input_scaler = joblib.load(INPUT_SCALER_FILE)
_target_scaler = joblib.load(TARGET_SCALER_FILE)


def correct(physics_output: dict) -> dict:
    """
    Apply the ML physics->CFD correction to one engine output dict (as
    produced by core.engine.run_engine).

    Returns a NEW dict: a copy of `physics_output` with the 7
    FEATURES replaced by their ML-corrected (CFD-like) values, plus
    the raw physics values preserved under a `physics_<field>` key and
    the predicted delta under a `delta_<field>` key, for transparency.
    """
    x = np.array([[float(physics_output[f]) for f in FEATURES]], dtype=float)

    x_scaled = _input_scaler.transform(x)
    delta_scaled = _model.predict(x_scaled)
    delta = _target_scaler.inverse_transform(delta_scaled).reshape(-1)

    corrected = dict(physics_output)
    for i, name in enumerate(FEATURES):
        physics_value = float(x[0, i])
        corrected[f"physics_{name}"] = physics_value
        corrected[f"delta_{name}"] = float(delta[i])
        corrected[name] = physics_value + float(delta[i])

    return corrected


def correct_batch(physics_rows) -> list:
    """Vectorised convenience wrapper over a list/iterable of dicts."""
    if not physics_rows:
        return []
    x = np.array(
        [[float(row[f]) for f in FEATURES] for row in physics_rows], dtype=float
    )
    x_scaled = _input_scaler.transform(x)
    delta_scaled = _model.predict(x_scaled)
    delta = _target_scaler.inverse_transform(delta_scaled)

    results = []
    for row, x_row, d_row in zip(physics_rows, x, delta):
        corrected = dict(row)
        for i, name in enumerate(FEATURES):
            corrected[f"physics_{name}"] = float(x_row[i])
            corrected[f"delta_{name}"] = float(d_row[i])
            corrected[name] = float(x_row[i]) + float(d_row[i])
        results.append(corrected)
    return results