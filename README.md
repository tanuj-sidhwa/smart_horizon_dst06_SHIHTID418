# Turbojet Design Framework (100N Thrust Class)

Requirements → Physics model → ML physics→CFD correction → NSGA-II multi-objective search → Pareto-optimal engine designs → CAD-ready geometry (FreeCAD).

---

## Problem Statement

> **Challenge ID:** SH-DST-06

**Context:**  
India lacks a fully indigenous, modular turbojet platform for UAVs/UCAVs and loitering munitions, resulting in high import dependency, elevated costs, and restricted IP control. There is a critical need to develop a physics-driven, GTRE-validated, test-ready turbojet engine to support defence applications with cost efficiency and export potential.

**Solution:**  
This framework provides an end-to-end computational design and multi-objective optimization pipeline. It rapidly generates, optimizes, and validates modular 100N thrust class micro-turbojet architectures to accelerate indigenous propulsion R&D.

---

## Description

An end-to-end conceptual design and multi-objective optimization pipeline for 100N thrust class micro-turbojet engines:

1. **`core/`** — 0D thermodynamic cycle model (inlet → compressor → combustor → turbine → nozzle) paired with a preliminary geometry/mass-sizing engine (`core/geometry.py`).
2. **`ml/`** — Physics-to-CFD correction model using `HistGradientBoostingRegressor` to predict $\Delta = \text{CFD} - \text{Physics}$. Reduces prediction error from 6–36% (physics-only) down to 0.06–2.5%.
3. **`optimizer/`** — Pure NumPy implementation of NSGA-II (`optimizer/nsga2.py`) executing a 10-variable thermodynamic design search (`optimizer/engine_optimizer.py`).
4. **`cad/`** — Parametric FreeCAD generator (`cad/freecad_generate.py`) constructing 3D STEP and `.FCStd` solids (centrifugal compressor, annular combustor, axial turbine, convergent nozzle).
5. **`app/`** — Interactive Streamlit UI ("JETFORGE") for requirements entry, optimization, Pareto-front exploration, and CFD validation.

### System Architecture

```text
   YOUR REQUIREMENTS                         CFD DATASET
 (target thrust, TSFC,      ┌──────────────┐  (9,207 cases)
  max exhaust temp, ...)    │              │        │
        │                   │   core/      │        │
        ▼                   │  (physics    │◄───────┘
┌─────────────────┐         │   model)     │   pipeline/validate_against_cfd.py
│  optimizer/      │────────►              │   → outputs/cfd_validation.csv
│  engine_optimizer│         └──────┬───────┘
│  (NSGA-II)       │                │
└────────┬─────────┘                ▼
         │                   ┌──────────────┐
         │                   │   ml/        │
         │                   │ correction_  │◄── ml/train_physics_correction.py
         │                   │  model.py    │
         │                   └──────┬───────┘
         │                          │
         ▼                          ▼
  outputs/pareto_front.csv   outputs/best_design.json
  outputs/best_design.json          │
         │                          ▼
         │                  cad/freecad_generate.py
         ▼                          ▼
  app/streamlit_app.py       outputs/turbojet.step
```

### Project Structure

```text
turbojet_framework/
├── core/            physics model + geometry sizing
├── ml/              physics → CFD correction model
├── optimizer/       NSGA-II implementation
├── pipeline/        CLI entry points (validate_against_cfd.py, run_optimization.py)
├── cad/             FreeCAD generator (freecad_generate.py)
├── app/             Streamlit UI (streamlit_app.py)
├── data/            9,207-case CFD dataset
├── tests/           smoke tests (pytest)
├── outputs/         generated CSV/JSON/STEP files
└── requirements.txt
```

### Known Limitations

- Geometry sizing provides a preliminary structural estimate; detailed FEA analysis is required for flight certification.
- The 0D physics model relies on the ML correction layer to capture radial and 3D aerodynamic losses.
- The CAD output serves as a conceptual envelope rather than a final production drawing.

---

## Installation

### Prerequisites

- Python 3.9+
- [FreeCAD](https://www.freecad.org/downloads.php) (required only for 3D CAD model generation)

### Setup

```bash
# 1. Clone & navigate to project root
cd turbojet_framework

# 2. Virtual environment setup
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt
pip install pytest               # optional, for testing
```

| Package | Version | Used For |
|---|---|---|
| `numpy` | `>=2.0,<2.3` | Physics model, NSGA-II, ML pipeline |
| `pandas` | `>=2.0,<3.0` | Data manipulation and reporting |
| `scikit-learn` | `>=1.5,<1.8` | ML correction model |
| `joblib` | `>=1.4,<2.0` | Model artifact persistence |
| `streamlit` | `>=1.30` | Interactive web dashboard |
| `plotly` | `>=5.18` | Pareto-front charts |

---

## Usage

```bash
# 1. Run unit tests
python -m pytest tests/ -q

# 2. Validate physics + ML against CFD dataset
python pipeline/validate_against_cfd.py

# 3. Retrain ML correction model (optional)
python ml/train_physics_correction.py

# 4. Run optimizer for 100N thrust class
python pipeline/run_optimization.py --thrust 100 --thrust-band 3 --max-exhaust-temp 1150 --min-exhaust-velocity 300 --max-exhaust-velocity 500 --population 80 --generations 100

# 5. Generate CAD geometry
freecadcmd cad/freecad_generate.py --design outputs/best_design.json --out outputs/turbojet.step

# 6. Launch Web Interface
streamlit run app/streamlit_app.py
```

### Key Outputs (`outputs/`)

- `cfd_validation.csv` — Full accuracy evaluation across dataset cases.
- `pareto_front.csv` — Trade-off set of non-dominated designs.
- `best_design.json` — Parameters and performance of the recommended engine.
- `turbojet.step` / `turbojet.FCStd` — Exported 3D CAD files.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Maintain parameter key consistency across `core/engine.py`, `data/cfd_dataset.csv`, `ml/model_config.json`, and `optimizer/engine_optimizer.py`.
3. Confirm all tests pass via `python -m pytest tests/ -q` before submitting a PR.


