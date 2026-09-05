"""
app/streamlit_app.py
=====================
Interactive UI for the turbojet design framework — "JETFORGE" cockpit edition.

Run it from the project root:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

Flow:
  1. Enter your performance requirements (target thrust, allowable
     bands, exhaust-temperature limit, etc.) in the sidebar "Mission
     Control" panel.
  2. Hit the big ignition button -> NSGA-II searches the full design
     space through physics + ML-correction and returns a Pareto front.
  3. Browse the Pareto front (interactive scatter + full table),
     inspect the recommended "best compromise" design in detail
     (design vector, physics vs ML-corrected performance, geometry,
     mass, thrust-to-weight).
  4. Download: Pareto front CSV, best-design JSON (feeds
     cad/freecad_generate.py directly), best-design one-row CSV.
  5. Optional: validate the physics+ML pipeline against the full CFD
     dataset from the "CFD Validation" tab.

All original functionality is untouched — this file only changes the
presentation layer (styling, animation, layout).
"""

import json
import sys
import time
import warnings
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from optimizer.engine_optimizer import Requirements, optimize, pareto_to_dataframe  # noqa: E402
from core.engine import INPUTS  # noqa: E402

st.set_page_config(
    page_title="JETFORGE — Turbojet Design Optimizer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =======================================================================
# THEME — fonts, colors, glow, HUD scanlines, animated background
# =======================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Share+Tech+Mono&display=swap');

    :root{
        --jf-cyan:#00e5ff;
        --jf-orange:#ff7a1a;
        --jf-flame:#ffb703;
        --jf-red:#ff3b3b;
        --jf-green:#39ff88;
        --jf-bg:#060a12;
        --jf-panel:#0c121d;
        --jf-panel-2:#101826;
        --jf-line:#1d2a3d;
    }

    html, body, [class*="css"]  { font-family: 'Share Tech Mono', monospace; }

    /* ---- animated deep-space / hangar backdrop ---- */
    .stApp{
        background:
            radial-gradient(circle at 15% 10%, rgba(0,229,255,0.07), transparent 40%),
            radial-gradient(circle at 85% 0%, rgba(255,122,26,0.08), transparent 45%),
            repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 3px),
            linear-gradient(180deg, #05070c 0%, #070b13 40%, #05070c 100%);
        background-attachment: fixed;
    }

    /* subtle animated scanline sweep over the whole app */
    .stApp::before{
        content:"";
        position:fixed; inset:0; pointer-events:none; z-index:0;
        background: linear-gradient(180deg, transparent 0%, rgba(0,229,255,0.035) 50%, transparent 100%);
        background-size: 100% 6px;
        animation: jf-scan 9s linear infinite;
        opacity:0.6;
    }
    @keyframes jf-scan{ 0%{background-position-y:0;} 100%{background-position-y:2000px;} }

    h1, h2, h3, h4 { font-family:'Orbitron', sans-serif !important; letter-spacing:0.5px; }

    /* ---- title banner ---- */
    .jf-title{
        font-family:'Orbitron', sans-serif; font-weight:900; font-size:2.4rem;
        background: linear-gradient(90deg, var(--jf-cyan), #7fffd4 30%, var(--jf-orange) 70%, var(--jf-flame));
        background-size:300% auto;
        -webkit-background-clip:text; background-clip:text; color:transparent;
        animation: jf-titleflow 6s linear infinite;
        text-shadow: 0 0 30px rgba(0,229,255,0.15);
        margin-bottom:0;
    }
    @keyframes jf-titleflow{ 0%{background-position:0% 50%;} 100%{background-position:300% 50%;} }
    .jf-subtitle{ color:#7d8ba3; font-size:0.95rem; letter-spacing:1px; text-transform:uppercase; }

    /* ---- HUD panel / card ---- */
    .jf-card{
        background: linear-gradient(180deg, var(--jf-panel), var(--jf-panel-2));
        border:1px solid var(--jf-line);
        border-radius:14px;
        padding:16px 18px;
        position:relative;
        box-shadow: 0 0 0 1px rgba(0,229,255,0.03), 0 8px 24px rgba(0,0,0,0.35);
    }
    .jf-card::before{
        content:""; position:absolute; top:0; left:14px; right:14px; height:2px;
        background:linear-gradient(90deg, transparent, var(--jf-cyan), transparent);
        opacity:0.6;
    }
    .jf-card-title{
        font-family:'Orbitron', sans-serif; font-size:0.78rem; letter-spacing:2px;
        color:var(--jf-cyan); text-transform:uppercase; margin-bottom:10px;
        display:flex; align-items:center; gap:8px;
    }
    .jf-dot{ width:8px; height:8px; border-radius:50%; background:var(--jf-green);
        box-shadow:0 0 8px var(--jf-green); animation: jf-pulse 1.6s ease-in-out infinite; }
    @keyframes jf-pulse{ 0%,100%{opacity:1; transform:scale(1);} 50%{opacity:0.4; transform:scale(0.7);} }

    /* ---- metric tiles ---- */
    div[data-testid="stMetric"]{
        background: linear-gradient(180deg, var(--jf-panel), #0a0f18);
        border:1px solid var(--jf-line);
        border-radius:12px; padding:12px 14px 8px 14px;
        transition: box-shadow .25s ease, transform .25s ease;
        box-shadow: inset 0 0 0 1px rgba(0,229,255,0.02);
    }
    div[data-testid="stMetric"]:hover{
        transform: translateY(-2px);
        box-shadow: 0 0 22px rgba(0,229,255,0.18), inset 0 0 0 1px rgba(0,229,255,0.08);
        border-color: rgba(0,229,255,0.35);
    }
    div[data-testid="stMetricLabel"]{ color:#7d8ba3 !important; letter-spacing:0.5px; }
    div[data-testid="stMetricValue"]{
        color: var(--jf-cyan) !important; font-family:'Orbitron', sans-serif !important;
        text-shadow:0 0 12px rgba(0,229,255,0.35);
    }

    /* ---- sidebar ---- */
    section[data-testid="stSidebar"]{
        background: linear-gradient(180deg, #070b12, #050810);
        border-right: 1px solid var(--jf-line);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3{
        color: var(--jf-orange) !important;
    }

    /* ignition button */
    div.stButton > button{
        font-family:'Orbitron', sans-serif; font-weight:700; letter-spacing:2px;
        background: linear-gradient(90deg, #ff3b3b, var(--jf-orange));
        border:1px solid rgba(255,180,80,0.5);
        color:#0a0a0a; border-radius:10px; padding:0.7em 1em;
        box-shadow: 0 0 18px rgba(255,122,26,0.45);
        transition: all .2s ease;
    }
    div.stButton > button:hover{
        box-shadow: 0 0 30px rgba(255,122,26,0.8), 0 0 60px rgba(255,59,59,0.3);
        transform: translateY(-1px) scale(1.01);
        color:#fff;
    }
    div.stButton > button:active{ transform: scale(0.98); }

    /* download buttons */
    div.stDownloadButton > button{
        background: linear-gradient(90deg, #0d2230, #0a3b46);
        border:1px solid var(--jf-cyan); color: var(--jf-cyan);
        font-family:'Orbitron', sans-serif; letter-spacing:1px; border-radius:10px;
    }
    div.stDownloadButton > button:hover{
        box-shadow: 0 0 18px rgba(0,229,255,0.55); color:#fff;
    }

    /* tabs */
    .stTabs [data-baseweb="tab-list"]{ gap:6px; border-bottom:1px solid var(--jf-line); }
    .stTabs [data-baseweb="tab"]{
        background: var(--jf-panel); border-radius:10px 10px 0 0;
        color:#7d8ba3; font-family:'Orbitron', sans-serif; font-size:0.8rem;
        letter-spacing:1px; padding:10px 16px; border:1px solid var(--jf-line); border-bottom:none;
    }
    .stTabs [aria-selected="true"]{
        color: var(--jf-cyan) !important;
        box-shadow: inset 0 -3px 0 var(--jf-cyan);
        background: var(--jf-panel-2);
    }

    /* bordered result containers */
    div[data-testid="stVerticalBlockBorderWrapper"]{
        border-color: var(--jf-line) !important;
        background: linear-gradient(180deg, var(--jf-panel), #080c14);
        box-shadow: 0 0 0 1px rgba(0,229,255,0.03), 0 6px 20px rgba(0,0,0,0.3);
        border-radius:14px !important;
    }

    /* status chip */
    .jf-chip{
        display:inline-flex; align-items:center; gap:8px;
        font-family:'Orbitron', sans-serif; font-size:0.78rem; letter-spacing:1.5px;
        padding:6px 14px; border-radius:999px; text-transform:uppercase;
    }
    .jf-chip-go{ background:rgba(57,255,136,0.1); color:var(--jf-green); border:1px solid rgba(57,255,136,0.4);
        box-shadow:0 0 14px rgba(57,255,136,0.25); }
    .jf-chip-nogo{ background:rgba(255,59,59,0.1); color:var(--jf-red); border:1px solid rgba(255,59,59,0.4);
        box-shadow:0 0 14px rgba(255,59,59,0.25); }

    /* progress console */
    .jf-console{
        font-family:'Share Tech Mono', monospace; background:#050810; border:1px solid var(--jf-line);
        border-radius:10px; padding:10px 14px; color: var(--jf-cyan); font-size:0.85rem;
        box-shadow: inset 0 0 20px rgba(0,229,255,0.05);
    }

    ::-webkit-scrollbar{ width:10px; height:10px;}
    ::-webkit-scrollbar-track{ background:#05070c;}
    ::-webkit-scrollbar-thumb{ background:linear-gradient(180deg, var(--jf-cyan), var(--jf-orange)); border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


# =======================================================================
# ANIMATED TURBOJET SVG — spinning compressor/turbine, flickering flame,
# shock-diamond exhaust, flowing airflow particles.
# =======================================================================
def turbojet_svg(height: int = 190, status: str = "idle") -> str:
    """Return an animated cutaway turbojet as inline SVG.
    status: 'idle' | 'running' | 'go' | 'nogo' — tweaks the glow color / speed.
    """
    if status == "running":
        spool_speed, turb_speed, flame_op, ring = "0.35s", "0.28s", "1", "#ffb703"
    elif status == "go":
        spool_speed, turb_speed, flame_op, ring = "0.5s", "0.4s", "0.95", "#39ff88"
    elif status == "nogo":
        spool_speed, turb_speed, flame_op, ring = "0.9s", "0.7s", "0.6", "#ff3b3b"
    else:
        spool_speed, turb_speed, flame_op, ring = "1.1s", "0.9s", "0.75", "#00e5ff"

    return f"""
    <svg viewBox="0 0 1000 220" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="flameGrad" cx="30%" cy="50%" r="75%">
          <stop offset="0%" stop-color="#ffffff"/>
          <stop offset="25%" stop-color="#fff2a8"/>
          <stop offset="55%" stop-color="#ffb703"/>
          <stop offset="80%" stop-color="#ff5f1a"/>
          <stop offset="100%" stop-color="#ff5f1a" stop-opacity="0"/>
        </radialGradient>
        <linearGradient id="bodyGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#2b3648"/>
          <stop offset="45%" stop-color="#1a2130"/>
          <stop offset="100%" stop-color="#0c1119"/>
        </linearGradient>
        <radialGradient id="coreGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stop-color="{ring}" stop-opacity="0.55"/>
          <stop offset="100%" stop-color="{ring}" stop-opacity="0"/>
        </radialGradient>
      </defs>

      <!-- ambient core glow -->
      <ellipse cx="500" cy="110" rx="480" ry="90" fill="url(#coreGlow)">
        <animate attributeName="rx" values="460;500;460" dur="3s" repeatCount="indefinite"/>
      </ellipse>

      <!-- fuselage -->
      <path d="M40,110 C40,70 120,58 220,58 L740,58 C820,58 860,80 900,95 L900,125
               C860,140 820,162 740,162 L220,162 C120,162 40,150 40,110 Z"
            fill="url(#bodyGrad)" stroke="{ring}" stroke-opacity="0.35" stroke-width="1.5"/>

      <!-- inlet cone -->
      <polygon points="40,110 110,86 110,134" fill="#141b28" stroke="{ring}" stroke-opacity="0.5"/>

      <!-- === COMPRESSOR (fast spin) === -->
      <g transform="translate(210,110)">
        <circle r="46" fill="#0d131d" stroke="{ring}" stroke-opacity="0.6"/>
        <g id="fan1">
          <g stroke="{ring}" stroke-width="6" stroke-linecap="round" opacity="0.9">
            <line x1="0" y1="0" x2="0" y2="-40"/>
            <line x1="0" y1="0" x2="28" y2="-28"/>
            <line x1="0" y1="0" x2="40" y2="0"/>
            <line x1="0" y1="0" x2="28" y2="28"/>
            <line x1="0" y1="0" x2="0" y2="40"/>
            <line x1="0" y1="0" x2="-28" y2="28"/>
            <line x1="0" y1="0" x2="-40" y2="0"/>
            <line x1="0" y1="0" x2="-28" y2="-28"/>
          </g>
          <animateTransform attributeName="transform" type="rotate" from="0" to="360"
                             dur="{spool_speed}" repeatCount="indefinite"/>
        </g>
        <circle r="8" fill="{ring}"/>
      </g>

      <!-- combustor -->
      <rect x="300" y="80" width="160" height="60" rx="14" fill="#141b28" stroke="{ring}" stroke-opacity="0.45"/>
      <ellipse cx="380" cy="110" rx="70" ry="22" fill="url(#flameGrad)" opacity="{flame_op}">
        <animate attributeName="rx" values="60;76;60" dur="0.25s" repeatCount="indefinite"/>
        <animate attributeName="opacity" values="0.6;1;0.7;{flame_op}" dur="0.4s" repeatCount="indefinite"/>
      </ellipse>

      <!-- === TURBINE (spins opposite, slightly slower) === -->
      <g transform="translate(560,110)">
        <circle r="40" fill="#0d131d" stroke="{ring}" stroke-opacity="0.6"/>
        <g id="fan2">
          <g stroke="{ring}" stroke-width="5" stroke-linecap="round" opacity="0.85">
            <line x1="0" y1="0" x2="0" y2="-34"/>
            <line x1="0" y1="0" x2="24" y2="-24"/>
            <line x1="0" y1="0" x2="34" y2="0"/>
            <line x1="0" y1="0" x2="24" y2="24"/>
            <line x1="0" y1="0" x2="0" y2="34"/>
            <line x1="0" y1="0" x2="-24" y2="24"/>
            <line x1="0" y1="0" x2="-34" y2="0"/>
            <line x1="0" y1="0" x2="-24" y2="-24"/>
          </g>
          <animateTransform attributeName="transform" type="rotate" from="360" to="0"
                             dur="{turb_speed}" repeatCount="indefinite"/>
        </g>
        <circle r="7" fill="{ring}"/>
      </g>

      <!-- nozzle -->
      <polygon points="650,72 830,92 830,128 650,148" fill="#141b28" stroke="{ring}" stroke-opacity="0.45"/>

      <!-- exhaust jet flame -->
      <g opacity="0.95">
        <polygon points="830,92 990,108 830,128" fill="url(#flameGrad)">
          <animate attributeName="points"
                    values="830,92 990,108 830,128;830,92 950,108 830,128;830,92 990,108 830,128"
                    dur="0.3s" repeatCount="indefinite"/>
        </polygon>
        <polygon points="830,100 930,108 830,118" fill="#ffffff" opacity="0.85">
          <animate attributeName="opacity" values="0.5;0.9;0.5" dur="0.18s" repeatCount="indefinite"/>
        </polygon>
      </g>

      <!-- shock diamonds -->
      <g stroke="#fff7cf" stroke-width="1.4" fill="none" opacity="0.6">
        <polygon points="855,108 868,98 881,108 868,118">
          <animate attributeName="opacity" values="0.2;0.8;0.2" dur="0.5s" repeatCount="indefinite"/>
        </polygon>
        <polygon points="895,108 906,100 917,108 906,116">
          <animate attributeName="opacity" values="0.8;0.2;0.8" dur="0.5s" repeatCount="indefinite"/>
        </polygon>
        <polygon points="932,108 941,101 950,108 941,115">
          <animate attributeName="opacity" values="0.2;0.8;0.2" dur="0.5s" repeatCount="indefinite"/>
        </polygon>
      </g>

      <!-- airflow particles streaming through the core -->
      <g fill="{ring}">
        <circle r="3">
          <animateMotion dur="1.4s" repeatCount="indefinite"
            path="M60,100 C160,90 260,100 360,108 C460,112 560,108 660,104 C760,100 830,104 880,108"/>
        </circle>
        <circle r="3">
          <animateMotion dur="1.4s" begin="0.35s" repeatCount="indefinite"
            path="M60,120 C160,116 260,116 360,112 C460,110 560,112 660,116 C760,120 830,116 880,112"/>
        </circle>
        <circle r="2.4">
          <animateMotion dur="1.4s" begin="0.7s" repeatCount="indefinite"
            path="M60,110 C160,104 260,108 360,110 C460,110 560,110 660,110 C760,110 830,110 880,110"/>
        </circle>
        <circle r="2.4">
          <animateMotion dur="1.4s" begin="1.05s" repeatCount="indefinite"
            path="M60,95 C160,88 260,96 360,104 C460,108 560,104 660,100 C760,96 830,102 880,108"/>
        </circle>
      </g>
    </svg>
    """


def status_banner(pct: int, label: str, color: str) -> str:
    return f"""
    <div class="jf-console">
      <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
        <span>{label}</span><span>{pct}%</span>
      </div>
      <div style="height:8px; background:#0a0f18; border-radius:6px; overflow:hidden; border:1px solid #1d2a3d;">
        <div style="height:100%; width:{pct}%; background:linear-gradient(90deg,{color},#fff7cf);
             box-shadow:0 0 12px {color}; transition:width .3s ease;"></div>
      </div>
    </div>
    """


# =======================================================================
# Header
# =======================================================================
st.markdown('<div class="jf-title">🚀 JETFORGE — KJ-66-Class Turbojet Design Optimizer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="jf-subtitle">Requirements → Physics model → ML physics→CFD correction → '
    'NSGA-II multi-objective search → Pareto-optimal designs → CAD-ready design vector</div>',
    unsafe_allow_html=True,
)
st.write("")

engine_slot = st.empty()
engine_slot.markdown(turbojet_svg(status="idle"), unsafe_allow_html=True)
st.write("")

# ---------------------------------------------------------------------
# Sidebar: requirements form
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="jf-card-title"><span class="jf-dot"></span> MISSION CONTROL</div>',
        unsafe_allow_html=True,
    )
    st.header("Performance requirements")

    target_thrust = st.number_input("Target thrust [N]", min_value=1.0, max_value=2000.0, value=100.0, step=1.0)
    thrust_band = st.number_input("Allowable thrust band ± [N]", min_value=0.1, max_value=200.0, value=3.0, step=0.5)

    max_exhaust_temp = st.number_input("Max exhaust temperature [K]", min_value=500.0, max_value=2000.0, value=1200.0, step=10.0)

    col1, col2 = st.columns(2)
    with col1:
        min_exhaust_vel = st.number_input("Min exhaust velocity [m/s]", min_value=0.0, max_value=1000.0, value=250.0, step=10.0)
    with col2:
        max_exhaust_vel = st.number_input("Max exhaust velocity [m/s]", min_value=0.0, max_value=1500.0, value=450.0, step=10.0)

    st.divider()
    st.subheader("⚙ Optimizer settings")
    population = st.slider("Population size", 20, 200, 80, step=10)
    generations = st.slider("Generations", 10, 300, 100, step=10)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

    st.write("")
    run_clicked = st.button("🔥 IGNITE — RUN OPTIMIZATION", type="primary", use_container_width=True)
    st.caption("Spools the NSGA-II search across the physics + ML-corrected design space.")

# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "pareto_df" not in st.session_state:
    st.session_state.pareto_df = None
    st.session_state.best = None
    st.session_state.req = None

if run_clicked:
    req = Requirements(
        target_thrust_N=target_thrust,
        thrust_band_N=thrust_band,
        max_exhaust_temp_K=max_exhaust_temp,
        min_exhaust_velocity_m_s=min_exhaust_vel,
        max_exhaust_velocity_m_s=max_exhaust_vel,
        population_size=population,
        generations=generations,
        random_seed=int(seed),
    )

    engine_slot.markdown(turbojet_svg(status="running"), unsafe_allow_html=True)
    console = st.empty()

    sequence = [
        (12, "IGNITION SEQUENCE START", "#ffb703"),
        (30, "COMPRESSOR SPOOLING UP", "#00e5ff"),
        (48, "FUEL / AIR MIX STABILIZING", "#ff7a1a"),
        (65, "COMBUSTION NOMINAL", "#ff5f1a"),
    ]
    for pct, label, color in sequence:
        console.markdown(status_banner(pct, label, color), unsafe_allow_html=True)
        time.sleep(0.25)

    console.markdown(status_banner(80, "NSGA-II MULTI-OBJECTIVE SEARCH RUNNING…", "#00e5ff"), unsafe_allow_html=True)
    with st.spinner("Evaluating designs through physics + ML correction..."):
        pareto, best = optimize(req, verbose=False)

    console.markdown(status_banner(100, "SEARCH COMPLETE", "#39ff88"), unsafe_allow_html=True)
    time.sleep(0.2)
    console.empty()

    st.session_state.pareto_df = pareto_to_dataframe(pareto, req)
    st.session_state.best = best
    st.session_state.req = req

    engine_slot.markdown(
        turbojet_svg(status="go" if best["feasible"] else "nogo"), unsafe_allow_html=True
    )
    if best["feasible"]:
        st.success(f"✅ Found {len(pareto)} Pareto-optimal designs. Recommended design is FEASIBLE.")
    else:
        st.warning(f"⚠ Found {len(pareto)} Pareto-optimal designs. Closest design is INFEASIBLE — widen your bands.")
elif st.session_state.best is not None:
    engine_slot.markdown(
        turbojet_svg(status="go" if st.session_state.best["feasible"] else "nogo"), unsafe_allow_html=True
    )

# plotly dark HUD template, reused by every chart below
PLOT_TEMPLATE = go.layout.Template(layout=dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#c9d6e8", family="Share Tech Mono, monospace"),
    xaxis=dict(gridcolor="#1d2a3d", zerolinecolor="#1d2a3d"),
    yaxis=dict(gridcolor="#1d2a3d", zerolinecolor="#1d2a3d"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    colorway=["#00e5ff", "#ff7a1a", "#39ff88", "#ffb703", "#ff3b3b"],
))

# ---------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------
tab_summary, tab_pareto, tab_design, tab_cfd = st.tabs(
    ["📊 BEST DESIGN", "🌐 PARETO FRONT", "🧩 DESIGN VECTOR", "✅ CFD VALIDATION"]
)

with tab_summary:
    if st.session_state.best is None:
        st.info("Set your requirements in **Mission Control** (sidebar) and hit **🔥 IGNITE**.")
    else:
        best = st.session_state.best
        corrected = best["corrected"]
        physics = best["physics"]

        chip = (
            '<span class="jf-chip jf-chip-go">🟢 FEASIBLE</span>'
            if best["feasible"]
            else '<span class="jf-chip jf-chip-nogo">🔴 INFEASIBLE — CLOSEST FOUND</span>'
        )
        st.markdown(f"### Recommended design {chip}", unsafe_allow_html=True)
        st.write("")

        with st.container(border=True):
            st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> LIVE PERFORMANCE READOUT</div>', unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Thrust (ML-corrected)", f"{corrected['thrust_N']:.2f} N")
            c2.metric("TSFC", f"{corrected['tsfc_kg_N_s']:.5f} kg/(N·s)")
            c3.metric("Thermal efficiency", f"{corrected['thermal_efficiency']*100:.2f} %")
            c4.metric("Thrust-to-weight", f"{physics['thrust_to_weight']:.2f}")
            c5.metric("Est. mass", f"{physics['total_mass_kg']:.2f} kg")

            c6, c7, c8 = st.columns(3)
            c6.metric("Exhaust temp", f"{corrected['exhaust_temp_K']:.1f} K")
            c7.metric("Exhaust velocity", f"{corrected['exhaust_velocity_m_s']:.1f} m/s")
            c8.metric("Fuel flow", f"{corrected['fuel_flow_kg_s']*1000:.3f} g/s")

        st.write("")
        colA, colB = st.columns(2)
        with colA:
            with st.container(border=True):
                st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> KEY DESIGN VARIABLES</div>', unsafe_allow_html=True)
                key_vars = [
                    "compressor_pressure_ratio", "compressor_efficiency",
                    "turbine_inlet_temp_K", "turbine_efficiency",
                    "combustor_air_fuel_ratio", "mass_flow_kg_s", "rpm",
                ]
                st.table(pd.DataFrame({"value": {k: best["design"][k] for k in key_vars}}))
        with colB:
            with st.container(border=True):
                st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> GEOMETRY (PRELIMINARY SIZING)</div>', unsafe_allow_html=True)
                geo_vars = [
                    "compressor_diameter_mm", "combustor_length_mm",
                    "combustor_outer_diameter_mm", "combustor_inner_diameter_mm",
                    "nozzle_exit_diameter_mm", "overall_length_mm", "overall_max_diameter_mm",
                ]
                st.table(pd.DataFrame({"value": {k: physics.get(k) for k in geo_vars}}))

        st.write("")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                "⬇ Download best_design.json (for FreeCAD generator)",
                data=json.dumps(
                    {
                        "requirements": st.session_state.req.__dict__,
                        "feasible": best["feasible"],
                        "design": best["design"],
                        "physics_performance": physics,
                        "ml_corrected_performance": corrected,
                    },
                    indent=2, default=str,
                ),
                file_name="best_design.json",
                mime="application/json",
                use_container_width=True,
            )
        with dcol2:
            report_row = dict(best["design"])
            for k, v in corrected.items():
                report_row[f"ml_corrected_{k}"] = v
            st.download_button(
                "⬇ Download best_design_report.csv",
                data=pd.DataFrame([report_row]).to_csv(index=False),
                file_name="best_design_report.csv",
                mime="text/csv",
                use_container_width=True,
            )

with tab_pareto:
    if st.session_state.pareto_df is None:
        st.info("Run the optimizer to see the Pareto front.")
    else:
        df = st.session_state.pareto_df
        st.markdown(f"**{len(df)} Pareto-optimal designs**")

        with st.container(border=True):
            st.markdown(
                '<div class="jf-card-title"><span class="jf-dot"></span> 📈 RESULT — PARETO FRONT (THRUST vs TSFC)</div>',
                unsafe_allow_html=True,
            )
            fig = px.scatter(
                df, x="thrust_N_corrected", y="tsfc_kg_N_s",
                color="thermal_efficiency", size="thrust_to_weight",
                hover_data=["exhaust_temp_K_corrected", "exhaust_velocity_m_s_corrected", "total_mass_kg"],
                labels={
                    "thrust_N_corrected": "Thrust [N] (ML-corrected)",
                    "tsfc_kg_N_s": "TSFC [kg/(N·s)]",
                    "thermal_efficiency": "Thermal efficiency",
                },
                color_continuous_scale=["#0a3b46", "#00e5ff", "#39ff88"],
                template=PLOT_TEMPLATE,
            )
            fig.update_traces(marker=dict(line=dict(width=1, color="#050810")))
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        with st.container(border=True):
            st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> FULL PARETO TABLE</div>', unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True, height=400)
            st.download_button(
                "⬇ Download pareto_front.csv",
                data=df.to_csv(index=False),
                file_name="pareto_front.csv",
                mime="text/csv",
            )

with tab_design:
    with st.container(border=True):
        st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> 19-FIELD DESIGN VECTOR SCHEMA</div>', unsafe_allow_html=True)
        st.markdown("Shared by the physics model, ML corrector, optimizer, and CAD generator:")
        st.code("\n".join(INPUTS), language="text")
    if st.session_state.best is not None:
        st.write("")
        with st.container(border=True):
            st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> RECOMMENDED DESIGN — FULL VECTOR</div>', unsafe_allow_html=True)
            st.json(st.session_state.best["design"])

with tab_cfd:
    st.markdown(
        "Runs the physics model + ML corrector against the full validated CFD dataset "
        "(`data/cfd_dataset.csv`) and reports mean absolute % error against actual CFD results."
    )
    run_cfd = st.button("🛰 Run CFD validation")
    if run_cfd:
        from pipeline.validate_against_cfd import run as run_validation

        with st.spinner("Running physics + ML correction on every CFD case..."):
            out_path = PROJECT_ROOT / "outputs" / "cfd_validation.csv"
            result = run_validation(PROJECT_ROOT / "data" / "cfd_dataset.csv", out_path)

        from ml.correction_model import FEATURES
        summary_rows = []
        for f in FEATURES:
            pe, me = f"physics_pct_error_{f}", f"ml_pct_error_{f}"
            if pe in result.columns:
                summary_rows.append({
                    "output": f,
                    "physics mean |% error|": result[pe].abs().mean(),
                    "ML-corrected mean |% error|": result[me].abs().mean(),
                })
        summary = pd.DataFrame(summary_rows)

        with st.container(border=True):
            st.markdown('<div class="jf-card-title"><span class="jf-dot"></span> VALIDATION SUMMARY</div>', unsafe_allow_html=True)
            st.dataframe(summary, use_container_width=True)

        st.write("")
        with st.container(border=True):
            st.markdown(
                '<div class="jf-card-title"><span class="jf-dot"></span> 📈 RESULT — PHYSICS vs ML-CORRECTED ERROR</div>',
                unsafe_allow_html=True,
            )
            fig = px.bar(
                summary.melt(id_vars="output", var_name="model", value_name="mean_abs_pct_error"),
                x="output", y="mean_abs_pct_error", color="model", barmode="group",
                template=PLOT_TEMPLATE,
                color_discrete_sequence=["#ff7a1a", "#00e5ff"],
            )
            fig.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.download_button(
                "⬇ Download full cfd_validation.csv",
                data=result.to_csv(index=False),
                file_name="cfd_validation.csv",
                mime="text/csv",
            )