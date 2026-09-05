"""
pipeline/run_optimization.py
=============================
End-to-end CLI: performance requirements -> NSGA-II design search ->
Pareto front CSV + recommended design JSON, ready for
cad/freecad_generate.py.

Example:
    python pipeline/run_optimization.py \\
        --thrust 100 --thrust-band 3 \\
        --max-exhaust-temp 1150 \\
        --min-exhaust-velocity 300 --max-exhaust-velocity 500 \\
        --population 80 --generations 100

Outputs (in outputs/, created if missing):
    pareto_front.csv     every non-dominated design found, full design
                          vector + corrected performance + geometry
    best_design.json     the single recommended "best compromise" design
                          (full design vector + physics + ML-corrected
                          performance + geometry) — this is the file
                          cad/freecad_generate.py reads.
    best_design_report.csv   the same, one row, for spreadsheet use.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")

from optimizer.engine_optimizer import Requirements, optimize, pareto_to_dataframe  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--thrust", type=float, default=100.0, help="Target thrust [N]")
    parser.add_argument("--thrust-band", type=float, default=3.0, help="+/- allowable band around target thrust [N]")
    parser.add_argument("--max-exhaust-temp", type=float, default=1200.0, help="Max exhaust temperature [K]")
    parser.add_argument("--min-exhaust-velocity", type=float, default=250.0, help="Min exhaust velocity [m/s]")
    parser.add_argument("--max-exhaust-velocity", type=float, default=450.0, help="Max exhaust velocity [m/s]")
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    req = Requirements(
        target_thrust_N=args.thrust,
        thrust_band_N=args.thrust_band,
        max_exhaust_temp_K=args.max_exhaust_temp,
        min_exhaust_velocity_m_s=args.min_exhaust_velocity,
        max_exhaust_velocity_m_s=args.max_exhaust_velocity,
        population_size=args.population,
        generations=args.generations,
        random_seed=args.seed,
    )

    print("Requirements:")
    print(f"  Target thrust:     {req.target_thrust_N} N  (band +/- {req.thrust_band_N} N)")
    print(f"  Max exhaust temp:  {req.max_exhaust_temp_K} K")
    print(f"  Exhaust velocity:  [{req.min_exhaust_velocity_m_s}, {req.max_exhaust_velocity_m_s}] m/s")
    print()

    pareto, best = optimize(req)

    if best is None:
        print("No individuals produced — check requirements/bounds.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pareto_to_dataframe(pareto, req)
    pareto_path = args.output_dir / "pareto_front.csv"
    df.to_csv(pareto_path, index=False)

    best_design_path = args.output_dir / "best_design.json"
    with open(best_design_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "requirements": {
                    "target_thrust_N": req.target_thrust_N,
                    "thrust_band_N": req.thrust_band_N,
                    "max_exhaust_temp_K": req.max_exhaust_temp_K,
                    "min_exhaust_velocity_m_s": req.min_exhaust_velocity_m_s,
                    "max_exhaust_velocity_m_s": req.max_exhaust_velocity_m_s,
                },
                "feasible": best["feasible"],
                "design": best["design"],
                "physics_performance": best["physics"],
                "ml_corrected_performance": best["corrected"],
            },
            f,
            indent=2,
        )

    import pandas as pd
    report_row = dict(best["design"])
    for k, v in best["corrected"].items():
        report_row[f"ml_corrected_{k}"] = v
    pd.DataFrame([report_row]).to_csv(args.output_dir / "best_design_report.csv", index=False)

    print(f"\nPareto front ({len(pareto)} designs) -> {pareto_path}")
    print(f"Recommended design -> {best_design_path}")
    print(f"\nRecommended design summary:")
    print(f"  Feasible:            {best['feasible']}")
    print(f"  Thrust (ML-corr.):   {best['corrected']['thrust_N']:.3f} N")
    print(f"  TSFC (ML-corr.):     {best['corrected']['tsfc_kg_N_s']:.6f} kg/(N.s)")
    print(f"  Thermal efficiency:  {best['corrected']['thermal_efficiency']:.4f}")
    print(f"  Exhaust temp:        {best['corrected']['exhaust_temp_K']:.1f} K")
    print(f"  Exhaust velocity:    {best['corrected']['exhaust_velocity_m_s']:.1f} m/s")
    print(f"  Total mass (est.):   {best['physics']['total_mass_kg']:.3f} kg")
    print(f"  Thrust-to-weight:    {best['physics']['thrust_to_weight']:.3f}")


if __name__ == "__main__":
    main()
