"""
ml/train_physics_correction.py
================================
Trains the physics -> CFD correction model.

    Delta = CFD - Physics          (7 performance features)
    R_hat = Physics + predicted Delta

This is a self-contained rewrite of the original training script: it
sources its (physics, CFD) pairs directly from
pipeline/validate_against_cfd.py's output (outputs/cfd_validation.csv)
instead of two hand-generated CSVs, so retraining is a two-command
round trip whenever the physics model or CFD dataset changes:

    python pipeline/validate_against_cfd.py
    python ml/train_physics_correction.py

Model selection: 4 candidate regressors (Ridge, RandomForest,
ExtraTrees, HistGradientBoosting) are trained on a 70% split, ranked
by validation-set correction RMSE, and the winner is retrained on
70%+15% (train+validation) and evaluated on the held-out 15% test set,
exactly as in the original training run (which selected
HistGradientBoosting — see ml/model_config.json).

Outputs (overwrites the bundled artifacts in ml/):
    correction_model.joblib, input_scaler.joblib, target_scaler.joblib,
    model_config.json, plus CSV reports in outputs/ml_training/.
"""

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES = [
    "thrust_N",
    "fuel_flow_kg_s",
    "tsfc_kg_N_s",
    "exhaust_velocity_m_s",
    "exhaust_temp_K",
    "compressor_power_W",
    "thermal_efficiency",
]

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent
REPORT_DIR = PROJECT_ROOT / "outputs" / "ml_training"


def load_physics_cfd_pairs(validation_csv: Path):
    if not validation_csv.exists():
        raise FileNotFoundError(
            f"{validation_csv} not found — run pipeline/validate_against_cfd.py first."
        )
    df = pd.read_csv(validation_csv)
    physics_cols = [f"physics_{f}" for f in FEATURES]
    cfd_cols = [f"cfd_{f}" for f in FEATURES]
    missing = [c for c in physics_cols + cfd_cols if c not in df.columns]
    if missing:
        raise ValueError(f"cfd_validation.csv missing expected columns: {missing}")

    physics = df[physics_cols].copy()
    physics.columns = FEATURES
    cfd = df[cfd_cols].copy()
    cfd.columns = FEATURES

    finite = np.isfinite(physics.to_numpy()).all(axis=1) & np.isfinite(cfd.to_numpy()).all(axis=1)
    return physics.loc[finite].reset_index(drop=True), cfd.loc[finite].reset_index(drop=True)


def calculate_metrics(y_true, y_pred, columns):
    rows = []
    for i, col in enumerate(columns):
        t, p = np.asarray(y_true)[:, i], np.asarray(y_pred)[:, i]
        rows.append({
            "output": col,
            "RMSE": float(np.sqrt(mean_squared_error(t, p))),
            "MAE": float(mean_absolute_error(t, p)),
            "R2": float(r2_score(t, p)),
        })
    t, p = np.asarray(y_true), np.asarray(y_pred)
    rows.append({
        "output": "OVERALL",
        "RMSE": float(np.sqrt(np.mean((t - p) ** 2))),
        "MAE": float(np.mean(np.abs(t - p))),
        "R2": np.nan,
    })
    return pd.DataFrame(rows)


def build_models():
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=1.0, n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=RANDOM_STATE,
            ),
            n_jobs=-1,
        ),
    }


def main():
    validation_csv = PROJECT_ROOT / "outputs" / "cfd_validation.csv"
    print(f"Loading physics/CFD pairs from {validation_csv} ...")
    physics, cfd = load_physics_cfd_pairs(validation_csv)
    print(f"Valid paired cases: {len(physics)}")

    correction = cfd - physics
    X, Y = physics.to_numpy(float), correction.to_numpy(float)
    n = len(X)

    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(n)
    rng.shuffle(idx)
    train_end, val_end = int(0.70 * n), int(0.85 * n)
    tr, va, te = idx[:train_end], idx[train_end:val_end], idx[val_end:]

    scaler_x, scaler_y = StandardScaler(), StandardScaler()
    Xtr_s = scaler_x.fit_transform(X[tr])
    Xva_s = scaler_x.transform(X[va])
    Ytr_s = scaler_y.fit_transform(Y[tr])

    print("\nBenchmarking candidate models on validation split...")
    results = []
    for name, model in build_models().items():
        m = clone(model)
        m.fit(Xtr_s, Ytr_s)
        pred_va = scaler_y.inverse_transform(m.predict(Xva_s))
        rmse = float(np.sqrt(mean_squared_error(Y[va], pred_va)))
        mae = float(mean_absolute_error(Y[va], pred_va))
        print(f"  {name:22s} val RMSE={rmse:.5g}  val MAE={mae:.5g}")
        results.append({"model": name, "correction_RMSE": rmse, "correction_MAE": mae})

    comparison = pd.DataFrame(results).sort_values("correction_RMSE").reset_index(drop=True)
    best_name = comparison.iloc[0]["model"]
    print(f"\nBest model: {best_name}")

    # retrain on train+val, evaluate on test
    X_trva, Y_trva = X[np.concatenate([tr, va])], Y[np.concatenate([tr, va])]
    final_x, final_y = StandardScaler(), StandardScaler()
    X_trva_s = final_x.fit_transform(X_trva)
    Y_trva_s = final_y.fit_transform(Y_trva)
    X_te_s = final_x.transform(X[te])

    best_model = clone(build_models()[best_name])
    best_model.fit(X_trva_s, Y_trva_s)
    pred_delta_te = final_y.inverse_transform(best_model.predict(X_te_s))
    R_pred_te = X[te] + pred_delta_te

    final_metrics = calculate_metrics(cfd.to_numpy(float)[te], R_pred_te, FEATURES)
    physics_baseline = calculate_metrics(cfd.to_numpy(float)[te], X[te], FEATURES)
    print("\nFinal (ML-corrected) test metrics:")
    print(final_metrics.to_string(index=False))
    print("\nPhysics-only baseline test metrics:")
    print(physics_baseline.to_string(index=False))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, OUTPUT_DIR / "correction_model.joblib")
    joblib.dump(final_x, OUTPUT_DIR / "input_scaler.joblib")
    joblib.dump(final_y, OUTPUT_DIR / "target_scaler.joblib")

    config = {
        "model_type": best_name,
        "input_type": "physics_model_outputs",
        "target_type": "absolute_cfd_correction",
        "correction_definition": "Delta = CFD - Physics",
        "prediction_definition": "R_hat = P + Delta_hat",
        "features": FEATURES,
        "number_of_cases_used": int(n),
        "random_state": RANDOM_STATE,
        "train_fraction": 0.70,
        "validation_fraction": 0.15,
        "test_fraction": 0.15,
    }
    with open(OUTPUT_DIR / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    comparison.to_csv(REPORT_DIR / "validation_model_comparison.csv", index=False)
    final_metrics.to_csv(REPORT_DIR / "test_metrics.csv", index=False)
    physics_baseline.to_csv(REPORT_DIR / "physics_only_baseline_metrics.csv", index=False)

    print(f"\nSaved model + scalers + config to {OUTPUT_DIR}")
    print(f"Saved training reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
