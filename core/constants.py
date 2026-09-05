"""
core.constants
================
Central physical / engineering constants used across every module in
`core`. Nothing here is design-specific — design choices belong in the
design-vector (see core.engine.INPUTS) or in geometry.py's sizing
assumptions, which are called out explicitly as approximations.
"""

# ------------------------------------------------------------------
# Working fluid (air, treated as calorically-perfect gas)
# ------------------------------------------------------------------
GAMMA = 1.4                # ratio of specific heats [-]
R = 287.058                # specific gas constant, air [J/(kg*K)]
CP = 1004.5                # specific heat at constant pressure, air [J/(kg*K)]

# ------------------------------------------------------------------
# Fuel (kerosene / Jet-A default)
# ------------------------------------------------------------------
FUEL_LHV = 43_000_000.0    # lower heating value [J/kg]

# ------------------------------------------------------------------
# Mechanical losses
# ------------------------------------------------------------------
MECHANICAL_EFFICIENCY = 0.98   # turbine-to-compressor shaft coupling

# ------------------------------------------------------------------
# Ambient / standard-day reference condition (sea level static)
# ------------------------------------------------------------------
AMBIENT_PRESSURE_PA = 101325.0
AMBIENT_TEMPERATURE_K = 288.15
STANDARD_GRAVITY = 9.80665

# ------------------------------------------------------------------
# Preliminary-sizing / mass-estimation constants (core.geometry)
# ------------------------------------------------------------------
# These are ENGINEERING APPROXIMATIONS used only to turn a converged
# thermodynamic design point into a self-consistent set of geometric
# and mass numbers for CAD scaffolding and the thrust/weight objective.
# They are intentionally simple (closed-form, no FEA/CFD) so they run
# inside an optimizer loop. Replace with a validated structural model
# before using the resulting mass/T:W numbers for certification-grade
# work.

DENSITY_COMPRESSOR_ALLOY_KG_M3 = 2810.0   # aluminium/Al-alloy rotor+casing
DENSITY_COMBUSTOR_ALLOY_KG_M3 = 8000.0    # Inconel/stainless liner+case
DENSITY_TURBINE_ALLOY_KG_M3 = 8220.0      # Inconel turbine disc/blades
DENSITY_NOZZLE_ALLOY_KG_M3 = 8000.0       # stainless exhaust cone

# Fraction of each section's bounding cylindrical shell that is
# actually solid material (rest is air/void/hollow passage)
COMPRESSOR_SOLIDITY_FRACTION = 0.35
COMBUSTOR_SOLIDITY_FRACTION = 0.12
TURBINE_SOLIDITY_FRACTION = 0.30
NOZZLE_SOLIDITY_FRACTION = 0.20

# Ancillary hardware (shaft, bearings, fuel system, casing bolts, etc.)
# as a fraction of the summed rotor+case mass estimate above
ANCILLARY_MASS_FRACTION = 0.20

# Lattice/infill heuristic (lightweight 3D-printed brackets only, NOT
# pressure-boundary parts): approximate surface-area-to-volume ratio
# contributed by a cubic lattice of given cell size & relative
# density. Coarse heuristic, not a homogenization result.
LATTICE_SA_V_COEFFICIENT = 6.0   # cube SA/V constant, dimensionless
