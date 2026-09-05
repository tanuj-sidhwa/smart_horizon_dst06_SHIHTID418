"""
Detailed parametric turbojet CAD generator for FreeCAD.

Replaces the old envelope-style generator.  The geometry is organized as a
small centrifugal-compressor / annular-combustor / single-stage-axial-turbine
engine, suitable for conceptual CAD, packaging and a hackathon demonstration.

IMPORTANT: this is a conceptual geometry generator, not a manufacturing or
CFD-validated blade design.  The optimizer JSON remains the single source of
truth for the dimensions it already provides; no NSGA-II variables are added.

Run inside FreeCAD:
    freecadcmd cad/freecad_generate.py \
      --design outputs/best_design.json \
      --out outputs/turbojet_final.step

Use --cutaway to export a casing with a longitudinal inspection window.  The
FCStd always contains separate groups/solids so the casing can also be hidden
in SolidWorks/FreeCAD to inspect the internals.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# Works both when FreeCAD executes this file directly (where __file__ exists)
# and when FreeCADCmd executes it through exec() (where __file__ is absent).
if "__file__" in globals():
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
else:
    PROJECT_ROOT = Path.cwd().resolve()
    # If launched from the project's parent directory, step into complete/.
    if not (PROJECT_ROOT / "outputs" / "best_design.json").exists() and (PROJECT_ROOT / "complete" / "outputs" / "best_design.json").exists():
        PROJECT_ROOT = PROJECT_ROOT / "complete"


def load_design(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "design" not in payload:
        raise KeyError("Design JSON must contain a top-level 'design' object")
    return payload["design"]


def rot_x(FreeCAD):
    # FreeCAD primitives are made along +Z; rotate +Z to +X.
    return FreeCAD.Rotation(FreeCAD.Vector(0, 1, 0), 90)


def place_x(FreeCAD, shape, x):
    shape.Placement = FreeCAD.Placement(FreeCAD.Vector(x, 0, 0), rot_x(FreeCAD))
    return shape


def xcyl(Part, FreeCAD, radius, length, x=0.0):
    return place_x(FreeCAD, Part.makeCylinder(radius, length), x)


def xcone(Part, FreeCAD, r1, r2, length, x=0.0):
    return place_x(FreeCAD, Part.makeCone(r1, r2, length), x)


def xring(Part, FreeCAD, outer_d, inner_d, length, x=0.0):
    ro = max(outer_d / 2.0, 0.1)
    ri = max(min(inner_d / 2.0, ro - 0.05), 0.0)
    outer = Part.makeCylinder(ro, length)
    if ri <= 0.0:
        sh = outer
    else:
        sh = outer.cut(Part.makeCylinder(ri, length))
    return place_x(FreeCAD, sh, x)


def add_obj(doc, group, name, shape, FreeCAD, props=None):
    obj = doc.addObject("Part::Feature", name)
    obj.Shape = shape
    group.addObject(obj)
    for k, v in (props or {}).items():
        safe = str(k).replace("-", "_")
        if isinstance(v, bool):
            obj.addProperty("App::PropertyBool", safe, "Design")
            setattr(obj, safe, v)
        elif isinstance(v, int):
            obj.addProperty("App::PropertyInteger", safe, "Design")
            setattr(obj, safe, v)
        elif isinstance(v, float):
            obj.addProperty("App::PropertyFloat", safe, "Design")
            setattr(obj, safe, v)
        else:
            obj.addProperty("App::PropertyString", safe, "Design")
            setattr(obj, safe, str(v))
    return obj


def color(obj, rgb):
    try:
        obj.ViewObject.ShapeColor = rgb
    except Exception:
        pass


def annular_cone(Part, FreeCAD, od1, od2, id1, id2, length, x):
    outer = Part.makeCone(od1 / 2.0, od2 / 2.0, length)
    inner = Part.makeCone(max(id1 / 2.0, 0.01), max(id2 / 2.0, 0.01), length)
    return place_x(FreeCAD, outer.cut(inner), x)


def radial_blade_face(Part, FreeCAD, r_root, r_tip, a1_root, a2_root,
                      a1_tip, a2_tip):
    """Curved radial blade planform in the Y-Z plane.

    Angles are degrees around the engine X axis. The four corners are at
    root/tip and leading/trailing angular positions. The face is later
    extruded axially, giving a thin 3-D blade with a controlled sweep.
    """
    def v(r, a_deg):
        a = math.radians(a_deg)
        return FreeCAD.Vector(0, r * math.cos(a), r * math.sin(a))
    pts = [
        v(r_root, a1_root),
        v(r_tip, a1_tip),
        v(r_tip, a2_tip),
        v(r_root, a2_root),
    ]
    wire = Part.makePolygon(pts + [pts[0]])
    return Part.Face(wire)


def curved_blade(Part, FreeCAD, x, axial_thickness, r_root, r_tip,
                 root_lead, root_trail, tip_lead, tip_trail):
    face = radial_blade_face(Part, FreeCAD, r_root, r_tip,
                             root_lead, root_trail, tip_lead, tip_trail)
    # The face is in the Y-Z plane at X=0; extrude in +X, then place at x.
    sh = face.extrude(FreeCAD.Vector(axial_thickness, 0, 0))
    sh.translate(FreeCAD.Vector(x, 0, 0))
    return sh


def blade_ring(Part, FreeCAD, x, count, axial_thickness,
               r_root, r_tip, root_chord, tip_chord,
               sweep_deg=20.0, stagger_deg=0.0):
    shapes = []
    # Each blade is made with its own angular sweep; rotate the entire blade
    # around X to distribute it uniformly.
    for i in range(count):
        # Keep chord angular width modest; blade length can never exceed the
        # actual annulus span r_tip-r_root.
        base = i * 360.0 / count + stagger_deg
        blade = curved_blade(
            Part, FreeCAD, x, axial_thickness, r_root, r_tip,
            base - root_chord / 2.0,
            base + root_chord / 2.0,
            base + sweep_deg - tip_chord / 2.0,
            base + sweep_deg + tip_chord / 2.0,
        )
        shapes.append(blade)
    return Part.makeCompound(shapes)


def radial_injector(Part, FreeCAD, x, radius_start, radius_end,
                    angle_deg, body_radius, tip_radius):
    a = math.radians(angle_deg)
    direction = FreeCAD.Vector(0, -math.cos(a), -math.sin(a))
    start = FreeCAD.Vector(x, radius_start * math.cos(a), radius_start * math.sin(a))
    length = max(radius_start - radius_end, 1.0)
    body = Part.makeCylinder(body_radius, length, start, direction)
    # Small nozzle tip at the inward end.
    end = start + direction * length
    tip = Part.makeCone(tip_radius, max(0.6 * tip_radius, 0.3), max(2.0, body_radius), end, direction)
    return body.fuse(tip)


def sector_shell(Part, FreeCAD, od, id_, length, x, keep_deg=300.0, start_deg=30.0):
    """Annular shell with a longitudinal cutaway window.

    Construct a rectangular radial profile in the X-Y plane and revolve it
    about X through keep_deg. This avoids fragile boolean wedge operations.
    """
    ro = od / 2.0
    ri = max(id_ / 2.0, 0.1)
    p = [
        FreeCAD.Vector(x, ri, 0),
        FreeCAD.Vector(x, ro, 0),
        FreeCAD.Vector(x + length, ro, 0),
        FreeCAD.Vector(x + length, ri, 0),
        FreeCAD.Vector(x, ri, 0),
    ]
    face = Part.Face(Part.makePolygon(p))
    sh = face.revolve(FreeCAD.Vector(x, 0, 0), FreeCAD.Vector(1, 0, 0), keep_deg)
    if start_deg:
        sh.rotate(FreeCAD.Vector(x, 0, 0), FreeCAD.Vector(1, 0, 0), start_deg)
    return sh


def add_flange(Part, FreeCAD, doc, group, name, x, outer_d, inner_d, thickness):
    sh = xring(Part, FreeCAD, outer_d, inner_d, thickness, x)
    return add_obj(doc, group, name, sh, FreeCAD,
                   {"OuterDiameterMM": float(outer_d), "InnerDiameterMM": float(inner_d)})


def build_model(design, FreeCAD, Part, cutaway=True):
    # FreeCAD 1.1 headless runner can expose an existing/active document
    # differently from the GUI.  Always obtain the returned document and
    # explicitly make it active; never assume ActiveDocument is non-null.
    doc = FreeCAD.newDocument("Turbojet_Final")
    if doc is None:
        doc = getattr(FreeCAD, "ActiveDocument", None)
    if doc is None:
        raise RuntimeError(
            "FreeCAD could not create a document in headless mode. "
            "Run the script with FreeCADCmd using the -c exec(open(...).read()) "
            "method described below."
        )
    try:
        FreeCAD.setActiveDocument(doc.Name)
    except Exception:
        pass

    # ------------------------- INPUTS ---------------------------------
    D = float(design["compressor_diameter_mm"])
    n_comp = max(6, int(round(float(design["compressor_blade_count"]))))
    comb_L = float(design["combustor_length_mm"])
    comb_OD = float(design["combustor_outer_diameter_mm"])
    comb_ID = float(design["combustor_inner_diameter_mm"])
    liner_t = max(0.8, float(design["combustor_liner_thickness_mm"]))
    n_inj = max(2, int(round(float(design["combustor_num_injectors"]))))
    n_turb = max(8, int(round(float(design["turbine_blade_count"]))))
    hub_tip = min(max(float(design["turbine_hub_tip_ratio"]), 0.55), 0.82)
    nozzle_D = float(design["nozzle_exit_diameter_mm"])
    rpm = float(design.get("rpm", 0.0))

    # ---------------------- STATIONS / SIZING -------------------------
    # D is treated as the compressor impeller tip diameter, not the casing OD.
    inlet_L = 0.34 * D
    comp_L = 0.42 * D
    diffuser_L = 0.18 * D
    transition_L = 0.18 * D
    x_inlet = 0.0
    x_comp = x_inlet + inlet_L
    x_diff = x_comp + comp_L
    x_trans = x_diff + diffuser_L
    x_comb = x_trans + transition_L
    x_turb = x_comb + comb_L

    # Keep turbine dimensions coherent with the combustor rather than making
    # blade span explode when nozzle diameter is small.
    turb_tip_D = min(0.86 * comb_ID, max(0.92 * nozzle_D, 0.62 * comb_ID))
    turb_tip_D = max(turb_tip_D, 0.72 * D)
    turb_tip_D = min(turb_tip_D, 0.90 * comb_OD)
    turb_tip_R = turb_tip_D / 2.0
    turb_hub_R = hub_tip * turb_tip_R
    turb_L = max(0.30 * turb_tip_D, 0.28 * D)
    x_noz = x_turb + turb_L
    noz_L = max(1.0 * nozzle_D, 0.75 * D)
    overall_L = x_noz + noz_L

    casing_OD = 1.12 * D
    inlet_OD = 1.16 * D
    diffuser_OD = 1.06 * D
    transition_OD = comb_OD

    # Compressor wheel geometry.
    imp_tip_R = 0.42 * D
    imp_hub_R = max(0.15 * D, 5.0)
    imp_axial = max(3.0, 0.065 * D)
    diffuser_inner_R = 0.41 * D
    diffuser_outer_R = 0.51 * D

    # Shaft and bearing geometry.
    shaft_R = max(0.075 * D, 3.0)
    bearing_OD = 2.15 * shaft_R
    bearing_ID = 1.30 * shaft_R
    bearing_L = max(5.0, 0.12 * D)

    # -------------------------- GROUPS --------------------------------
    names = ["Inlet", "Compressor", "Diffuser", "ShaftSystem",
             "Combustor", "Turbine", "Exhaust", "Casing", "DesignData"]
    groups = {n: doc.addObject("App::DocumentObjectGroup", n) for n in names}

    # --------------------------- INLET --------------------------------
    inlet_wall = 2.0
    inlet_inner = max(D * 0.92, 1.0)
    inlet_outer = Part.makeCone(inlet_OD / 2.0, casing_OD / 2.0, inlet_L)
    inlet_inner_sh = Part.makeCone(max(inlet_OD / 2.0 - inlet_wall, inlet_inner / 2.0),
                                   max(casing_OD / 2.0 - inlet_wall, D * 0.45), inlet_L)
    inlet_shell = place_x(FreeCAD, inlet_outer.cut(inlet_inner_sh), x_inlet)
    add_obj(doc, groups["Inlet"], "InletCasing", inlet_shell, FreeCAD,
            {"InletOuterDiameterMM": inlet_OD, "LengthMM": inlet_L})
    add_flange(Part, FreeCAD, doc, groups["Inlet"], "InletLip",
               x_inlet, inlet_OD + 3.0, inlet_inner + 2.0, 3.0)

    # ------------------------- COMPRESSOR -----------------------------
    comp_case = xring(Part, FreeCAD, casing_OD, 1.04 * D, comp_L, x_comp)
    add_obj(doc, groups["Compressor"], "CompressorCasing", comp_case, FreeCAD,
            {"OuterDiameterMM": casing_OD, "FlowDiameterMM": 0.90 * D})

    backplate = xcyl(Part, FreeCAD, imp_tip_R, imp_axial, x_comp + comp_L - imp_axial - 2.0)
    add_obj(doc, groups["Compressor"], "CompressorBackplate", backplate, FreeCAD)

    hub = xcone(Part, FreeCAD, imp_hub_R, 0.22 * D, max(0.45 * comp_L, 10.0), x_comp + 0.18 * comp_L)
    add_obj(doc, groups["Compressor"], "ImpellerHub", hub, FreeCAD)

    eye_outer_R = 0.24 * D
    eye_inner_R = max(shaft_R * 1.25, 0.11 * D)
    eye = xring(Part, FreeCAD, 2.0 * eye_outer_R, 2.0 * eye_inner_R,
                max(0.18 * comp_L, 5.0), x_comp + 0.02 * comp_L)
    add_obj(doc, groups["Compressor"], "ImpellerEye", eye, FreeCAD)

    # Curved radial impeller blades: span is always <= imp_tip_R-imp_hub_R.
    comp_blades = blade_ring(
        Part, FreeCAD, x_comp + 0.24 * comp_L, n_comp, imp_axial,
        imp_hub_R, imp_tip_R,
        root_chord=12.0, tip_chord=7.0,
        sweep_deg=28.0, stagger_deg=0.0)
    add_obj(doc, groups["Compressor"], "CompressorImpellerBlades", comp_blades, FreeCAD,
            {"BladeCount": n_comp, "RootRadiusMM": imp_hub_R, "TipRadiusMM": imp_tip_R})

    # Diffuser vanes are outside the impeller and inside the compressor case.
    diff_vanes = blade_ring(
        Part, FreeCAD, x_diff + 0.10 * diffuser_L, n_comp, max(1.5, 0.05 * D),
        diffuser_inner_R, diffuser_outer_R,
        root_chord=9.0, tip_chord=5.0,
        sweep_deg=-10.0, stagger_deg=180.0 / n_comp)
    add_obj(doc, groups["Diffuser"], "DiffuserVanes", diff_vanes, FreeCAD,
            {"VaneCount": n_comp, "InnerRadiusMM": diffuser_inner_R, "OuterRadiusMM": diffuser_outer_R})
    diffuser_case = xring(Part, FreeCAD, diffuser_OD, 1.02 * D, diffuser_L, x_diff)
    add_obj(doc, groups["Diffuser"], "DiffuserCasing", diffuser_case, FreeCAD)

    # ------------------------- SHAFT ---------------------------------
    shaft = xcyl(Part, FreeCAD, shaft_R, overall_L * 0.88, 0.06 * overall_L)
    add_obj(doc, groups["ShaftSystem"], "MainShaft", shaft, FreeCAD,
            {"DiameterMM": 2 * shaft_R, "RPM": rpm})
    for i, bx in enumerate((x_comp + 0.06 * comp_L, x_turb + 0.62 * turb_L), 1):
        bearing = xring(Part, FreeCAD, bearing_OD, bearing_ID, bearing_L, bx)
        add_obj(doc, groups["ShaftSystem"], "Bearing_%d" % i, bearing, FreeCAD,
                {"ShaftDiameterMM": 2 * shaft_R})
    compressor_nut = xcyl(Part, FreeCAD, 1.35 * shaft_R, max(3.0, 0.07 * D),
                          x_comp + 0.06 * comp_L)
    add_obj(doc, groups["ShaftSystem"], "CompressorNut", compressor_nut, FreeCAD)

    # ------------------------ COMBUSTOR ------------------------------
    # Casing and liner are annular/open at both axial ends. No solid disk is
    # placed across the flow path, so the combustor is not incorrectly sealed.
    case_wall = max(2.0, 0.025 * comb_OD)
    case_ID = comb_OD - 2.0 * case_wall
    casing = xring(Part, FreeCAD, comb_OD, case_ID, comb_L, x_comb)
    add_obj(doc, groups["Combustor"], "CombustorOuterCasing", casing, FreeCAD,
            {"OuterDiameterMM": comb_OD, "InnerDiameterMM": case_ID})

    liner_OD = comb_ID + 2.0 * liner_t
    liner = xring(Part, FreeCAD, liner_OD, comb_ID, 0.92 * comb_L,
                  x_comb + 0.04 * comb_L)
    add_obj(doc, groups["Combustor"], "CombustorLiner", liner, FreeCAD,
            {"InnerDiameterMM": comb_ID, "WallThicknessMM": liner_t})

    # Front and rear annular support rings: these touch the liner/casing but
    # leave the central passage open.
    ring_t = max(2.0, 0.035 * comb_L)
    front_ring = xring(Part, FreeCAD, case_ID, liner_OD + 2.0, ring_t, x_comb)
    rear_ring = xring(Part, FreeCAD, case_ID, liner_OD + 2.0, ring_t,
                      x_comb + comb_L - ring_t)
    add_obj(doc, groups["Combustor"], "CombustorFrontSupport", front_ring, FreeCAD)
    add_obj(doc, groups["Combustor"], "CombustorRearSupport", rear_ring, FreeCAD)

    # Injector bosses and inward fuel nozzles, equally spaced around the axis.
    inj_x = x_comb + 0.24 * comb_L
    for i in range(n_inj):
        a = 360.0 * i / n_inj
        boss = radial_injector(Part, FreeCAD, inj_x, comb_OD / 2.0 + 1.0,
                               liner_OD / 2.0 + 2.0, a,
                               max(2.0, 0.028 * comb_OD),
                               max(1.2, 0.015 * comb_OD))
        add_obj(doc, groups["Combustor"], "FuelInjector_%02d" % (i + 1), boss, FreeCAD,
                {"InjectorNumber": i + 1})

    # Igniter boss on the side of the liner.
    ign_x = x_comb + 0.56 * comb_L
    ign = radial_injector(Part, FreeCAD, ign_x, comb_OD / 2.0 + 1.0,
                          liner_OD / 2.0 + 4.0, 90.0, 2.8, 2.0)
    add_obj(doc, groups["Combustor"], "IgniterBoss", ign, FreeCAD)

    # Simple flame-holder ring with spokes, not a sealed bulkhead.
    fh_x = x_comb + 0.72 * comb_L
    fh = xring(Part, FreeCAD, liner_OD + 8.0, comb_ID - 4.0,
               max(2.0, 0.025 * comb_L), fh_x)
    add_obj(doc, groups["Combustor"], "FlameHolderRing", fh, FreeCAD)

    # -------------------------- TRANSITION ----------------------------
    # Annular diffuser/transition from combustor liner region to turbine.
    trans_len = transition_L
    trans = annular_cone(Part, FreeCAD,
                         comb_ID, turb_tip_D * 1.04,
                         0.90 * comb_ID, 2.0 * turb_hub_R,
                         trans_len, x_trans)
    add_obj(doc, groups["Diffuser"], "CombustorToTurbineTransition", trans, FreeCAD)

    # --------------------------- TURBINE ------------------------------
    stator_x = x_turb + 0.08 * turb_L
    rotor_x = x_turb + 0.48 * turb_L
    stator_OD = turb_tip_D + 5.0
    stator_ID = 2.0 * turb_hub_R * 1.02
    stator_ring = xring(Part, FreeCAD, stator_OD, stator_ID,
                        max(3.0, 0.15 * turb_L), stator_x)
    add_obj(doc, groups["Turbine"], "TurbineStatorRing", stator_ring, FreeCAD)

    stator = blade_ring(
        Part, FreeCAD, stator_x, n_turb, max(1.5, 0.06 * D),
        turb_hub_R * 1.03, turb_tip_R * 0.96,
        root_chord=10.0, tip_chord=6.0,
        sweep_deg=12.0, stagger_deg=0.0)
    add_obj(doc, groups["Turbine"], "TurbineNozzleGuideVanes", stator, FreeCAD,
            {"VaneCount": n_turb})

    disk_t = max(3.0, 0.08 * turb_tip_D)
    disk = xcyl(Part, FreeCAD, turb_hub_R * 1.05, disk_t, rotor_x)
    add_obj(doc, groups["Turbine"], "TurbineRotorDisk", disk, FreeCAD,
            {"DiameterMM": 2.0 * turb_hub_R * 1.05})

    rotor = blade_ring(
        Part, FreeCAD, rotor_x, n_turb, max(1.8, 0.065 * D),
        turb_hub_R, turb_tip_R * 0.965,
        root_chord=8.0, tip_chord=4.5,
        sweep_deg=-18.0, stagger_deg=180.0 / n_turb)
    add_obj(doc, groups["Turbine"], "TurbineRotorBlades", rotor, FreeCAD,
            {"BladeCount": n_turb, "HubRadiusMM": turb_hub_R, "TipRadiusMM": turb_tip_R})

    turbine_case = xring(Part, FreeCAD, stator_OD, 0.88 * turb_tip_D,
                         0.85 * turb_L, x_turb)
    add_obj(doc, groups["Turbine"], "TurbineCasing", turbine_case, FreeCAD)

    # -------------------------- EXHAUST -------------------------------
    collector_L = max(0.22 * D, 8.0)
    collector = annular_cone(Part, FreeCAD,
                             stator_OD, 1.30 * nozzle_D,
                             2.0 * shaft_R, 1.12 * nozzle_D,
                             collector_L, x_noz)
    add_obj(doc, groups["Exhaust"], "TurbineExhaustCollector", collector, FreeCAD)

    nozzle_x = x_noz + collector_L
    nozzle_L_actual = max(noz_L - collector_L, 0.65 * D)
    nozzle_outer = Part.makeCone(0.65 * nozzle_D, nozzle_D / 2.0, nozzle_L_actual)
    nozzle_inner = Part.makeCone(max(0.65 * nozzle_D - 2.5, nozzle_D / 2.0 + 0.8),
                                 max(nozzle_D / 2.0 - 2.0, 1.0), nozzle_L_actual)
    nozzle = place_x(FreeCAD, nozzle_outer.cut(nozzle_inner), nozzle_x)
    add_obj(doc, groups["Exhaust"], "ConvergentExhaustNozzle", nozzle, FreeCAD,
            {"ExitDiameterMM": nozzle_D})

    tail_len = max(0.45 * D, 15.0)
    tail_start = x_turb + 0.74 * turb_L
    tail = xcone(Part, FreeCAD, turb_hub_R * 0.92, max(0.20 * nozzle_D, 3.0),
                 tail_len, tail_start)
    add_obj(doc, groups["Exhaust"], "ExhaustTailCone", tail, FreeCAD)

    centerbody = xcone(Part, FreeCAD, max(0.18 * nozzle_D, 3.0),
                       max(0.10 * nozzle_D, 2.0),
                       0.75 * nozzle_L_actual, nozzle_x + 0.05 * nozzle_L_actual)
    add_obj(doc, groups["Exhaust"], "NozzleCenterbody", centerbody, FreeCAD)

    # --------------------------- CASING -------------------------------
    # Full casing sections are separate from internal hardware.  If cutaway
    # is enabled, each shell retains 300 degrees and leaves a 60-degree window.
    shell_keep = 300.0 if cutaway else 360.0
    shell_start = 30.0
    shell_specs = [
        ("InletOuterShell", x_inlet, inlet_L, inlet_OD, inlet_inner),
        ("CompressorOuterShell", x_comp, comp_L, casing_OD, 1.04 * D),
        ("DiffuserOuterShell", x_diff, diffuser_L, diffuser_OD, 1.02 * D),
        ("CombustorOuterShell", x_comb, comb_L, comb_OD, case_ID),
        ("TurbineOuterShell", x_turb, turb_L, stator_OD, 0.88 * turb_tip_D),
    ]
    for name, sx, sl, od, iid in shell_specs:
        if cutaway:
            sh = sector_shell(Part, FreeCAD, od, iid, sl, sx, shell_keep, shell_start)
        else:
            sh = xring(Part, FreeCAD, od, iid, sl, sx)
        add_obj(doc, groups["Casing"], name, sh, FreeCAD,
                {"OuterDiameterMM": od, "Cutaway": cutaway})

    # Structural/mount flanges around casing joints.
    for j, fx in enumerate((x_comp, x_diff, x_comb, x_turb, nozzle_x), 1):
        fd = comb_OD if fx >= x_comb and fx <= x_turb else max(casing_OD, inlet_OD)
        flange = xring(Part, FreeCAD, fd * 1.03, fd * 0.88, 3.0, fx)
        add_obj(doc, groups["Casing"], "MountFlange_%02d" % j, flange, FreeCAD)

    # -------------------------- DESIGN DATA ---------------------------
    meta = doc.addObject("App::FeaturePython", "OptimizedDesign")
    groups["DesignData"].addObject(meta)
    for key, value in sorted(design.items()):
        if value is None:
            continue
        safe = key.replace("-", "_")
        if isinstance(value, int):
            meta.addProperty("App::PropertyInteger", safe, "OptimizedDesign")
            setattr(meta, safe, value)
        elif isinstance(value, (float, int)):
            meta.addProperty("App::PropertyFloat", safe, "OptimizedDesign")
            setattr(meta, safe, float(value))
        else:
            meta.addProperty("App::PropertyString", safe, "OptimizedDesign")
            setattr(meta, safe, str(value))
    meta.addProperty("App::PropertyString", "Architecture", "CAD")
    meta.Architecture = "Centrifugal compressor + annular combustor + single-stage axial turbine"
    meta.addProperty("App::PropertyBool", "Cutaway", "CAD")
    meta.Cutaway = cutaway
    meta.addProperty("App::PropertyFloat", "OverallLengthMM", "CAD")
    meta.OverallLengthMM = overall_L
    meta.addProperty("App::PropertyFloat", "OverallMaxDiameterMM", "CAD")
    meta.OverallMaxDiameterMM = max(inlet_OD, comb_OD, stator_OD)
    meta.addProperty("App::PropertyFloat", "TurbineTipDiameterMM", "CAD")
    meta.TurbineTipDiameterMM = turb_tip_D

    # --------------------------- COLORS -------------------------------
    palette = {
        "Inlet": (0.68, 0.70, 0.74),
        "Compressor": (0.55, 0.65, 0.82),
        "Diffuser": (0.48, 0.58, 0.72),
        "ShaftSystem": (0.72, 0.72, 0.74),
        "Combustor": (0.78, 0.48, 0.22),
        "Turbine": (0.52, 0.54, 0.58),
        "Exhaust": (0.60, 0.61, 0.64),
        "Casing": (0.38, 0.40, 0.44),
    }
    for gname, group in groups.items():
        if gname not in palette:
            continue
        for obj in group.Group:
            color(obj, palette[gname])

    doc.recompute()
    print("\nTurbojet CAD generated")
    print("  Architecture: centrifugal compressor / annular combustor / single-stage axial turbine")
    print("  Overall length: %.1f mm" % overall_L)
    print("  Maximum OD:     %.1f mm" % max(inlet_OD, comb_OD, stator_OD))
    print("  Compressor:     %d blades" % n_comp)
    print("  Injectors:      %d" % n_inj)
    print("  Turbine:        %d rotor blades + %d guide vanes" % (n_turb, n_turb))
    print("  Turbine tip:    %.1f mm" % turb_tip_D)
    print("  Nozzle exit:    %.1f mm" % nozzle_D)
    print("  Cutaway:        %s" % cutaway)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--design", type=Path, default=PROJECT_ROOT / "outputs" / "best_design.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs" / "turbojet_final.step")
    parser.add_argument("--fcstd", type=Path, default=None)
    parser.add_argument("--no-cutaway", action="store_true",
                        help="Use full 360-degree casing instead of a 60-degree inspection window")
    args = parser.parse_args()

    try:
        import FreeCAD
        import Part
    except ImportError:
        sys.exit("ERROR: run this file with FreeCAD's Python/freecadcmd.")

    design = load_design(args.design)
    doc = build_model(design, FreeCAD, Part, cutaway=not args.no_cutaway)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shapes = []
    for obj in doc.Objects:
        if hasattr(obj, "Shape") and not obj.Shape.isNull():
            shapes.append(obj.Shape)
    Part.makeCompound(shapes).exportStep(str(args.out))
    print("Saved STEP: %s" % args.out)

    fcstd = args.fcstd or args.out.with_suffix(".FCStd")
    doc.saveAs(str(fcstd))
    print("Saved FCStd: %s" % fcstd)


# FreeCADCmd 1.1 may import .py files instead of executing them as __main__.
# The command-line runner therefore needs an explicit execution path.
if __name__ == "__main__":
    main()
