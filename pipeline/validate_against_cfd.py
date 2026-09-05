"""
pipeline/validate_against_cfd.py
=================================
Runs the physics model (and the ML physics->CFD corrector) on every
input row of the CFD dataset, and writes ONE clean CSV containing:

    - every original CFD design-vector input column
    - every physics-model-predicted performance/geometry output
      (prefixed `physics_`)
    - every ML-corrected performance output (prefixed `ml_`)
    - every actual CFD performance output (prefixed `cfd_`)
    - per-row absolute/percent error for physics-vs-CFD and
      ML-vs-CFD, for the 7 performance features

This is the single artifact that answers "how good is the physics
model, and how much does the ML corrector help" — no hard-coded paths,
run from the project root:

    python pipeline/validate_against_cfd.py
    python pipeline/validate_against_cfd.py --input data/cfd_dataset.csv --output outputs/cfd_validation.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.engine import run_engine, INPUTS          # noqa: E402
from ml.correction_model import correct, FEATURES   # noqa: E402


def run(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    missing_inputs = [c for c in INPUTS if c not in df.columns]
    if missing_inputs:
        raise ValueError(f"Input CSV is missing design columns: {missing_inputs}")

    rows = []
    n_failed = 0
    for idx, r in df.iterrows():
        try:
            design = {c: float(r[c]) for c in INPUTS}
            physics_out = run_engine(**design)
            ml_out = correct(physics_out)
        except Exception as exc:  # noqa: BLE001 - we want to keep going and report
            n_failed += 1
            print(f"  [row {idx}] FAILED: {exc}", file=sys.stderr)
            continue

        row = dict(design)

        for f in physics_out:
            if f not in INPUTS:
                row[f"physics_{f}"] = physics_out[f]

        for f in FEATURES:
            row[f"ml_{f}"] = ml_out[f]

        for f in FEATURES:
            if f in df.columns:
                cfd_actual = float(r[f])
                physics_val = float(physics_out[f])
                ml_val = float(ml_out[f])
                row[f"cfd_{f}"] = cfd_actual
                row[f"physics_error_{f}"] = physics_val - cfd_actual
                row[f"ml_error_{f}"] = ml_val - cfd_actual
                if cfd_actual != 0:
                    row[f"physics_pct_error_{f}"] = 100.0 * (physics_val - cfd_actual) / abs(cfd_actual)
                    row[f"ml_pct_error_{f}"] = 100.0 * (ml_val - cfd_actual) / abs(cfd_actual)

        rows.append(row)

    result = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)

    print(f"\nProcessed {len(df)} CFD rows -> {len(result)} succeeded, {n_failed} failed.")
    print(f"Saved: {output_csv}")

    # Quick summary of physics-vs-ML-vs-CFD agreement
    summary_rows = []
    for f in FEATURES:
        pe_col, me_col = f"physics_pct_error_{f}", f"ml_pct_error_{f}"
        if pe_col in result.columns:
            summary_rows.append({
                "output": f,
                "physics_mean_abs_pct_error": result[pe_col].abs().mean(),
                "ml_mean_abs_pct_error": result[me_col].abs().mean(),
            })
    summary = pd.DataFrame(summary_rows)
    print("\nMean absolute % error vs CFD:")
    print(summary.to_string(index=False))
    summary.to_csv(output_csv.with_name(output_csv.stem + "_summary.csv"), index=False)

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data" / "cfd_dataset.csv",
        help="CFD dataset CSV with the design-vector input columns.",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "outputs" / "cfd_validation.csv",
        help="Where to write the combined physics+ML+CFD comparison CSV.",
    )
    args = parser.parse_args()
    run(args.input, args.output)


if __name__ == "__main__":
    main()
