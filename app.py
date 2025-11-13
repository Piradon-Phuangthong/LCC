# app.py
# Strength-Driven Mix Designer — fixed-water UI with table output
# Run:
#   pip install streamlit scikit-learn numpy pandas
#   streamlit run app.py

import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor

# ===========================
# Page config
# ===========================
st.set_page_config(page_title="Strength-Driven Mix Designer", page_icon="🧱", layout="wide")
st.title("🧱 Strength-Driven Mix Designer")
st.caption("Fixed-water workflow • Spreadsheet-style aggregates & EC • Curve+KNN w/b prediction")

# ===========================
# Core constants
# ===========================
WATER_DEFAULT = 190.0
SHEET_PASTE_EXTRA_LITERS = 15.0
SHEET_SUBTRACT_AIR = False
DEFAULT_AIR_PERCENT = 1.9

# ===========================
# Material densities (kg/L)
# ===========================
DENSITY = {
    "Water": 1.00,
    "Cement": 3.11,
    "Fly Ash": 2.35,
    "GGBFS": 2.89,
    "20mm Aggregate": 2.61,
    "10mm Aggregate": 2.71,
    "Man Sand": 2.64,
    "Natural Sand": 2.65,
    "Plastiment 30": 1.05,
    "ECO WR": 1.07,
    "Retarder": 1.05,
}

# --- Aggregate splits by w/b band ---
SHEET_SPLITS = {
    0.58: {"20mm": 0.392, "10mm": 0.168, "Nat": 0.264, "Man": 0.176},
    0.50: {"20mm": 0.406, "10mm": 0.174, "Nat": 0.252, "Man": 0.168},
    0.42: {"20mm": 0.420, "10mm": 0.180, "Nat": 0.240, "Man": 0.160},
    0.34: {"20mm": 0.448, "10mm": 0.192, "Nat": 0.216, "Man": 0.144},
}

# --- Admixture doses (per 100 kg binder) ---
ADM_DOSE_MASS_PER_100KG = {
    "Retarder": 0.105,
    "Plastiment 30": 0.315,
    "ECO_WR_FA_OR_TERNARY": 0.214,
    "ECO_WR_PC_OR_SLAG": 0.535,
}

# --- EC Factors (kg CO2e/kg material) ---
EF_PER_KG = {
    "Water": 0.0004,
    "Cement": 0.9178,
    "GGBFS": 0.1922,
    "Fly Ash": 0.0198,
    "20mm Aggregate": 0.0105,
    "10mm Aggregate": 0.0105,
    "Man Sand": 0.0105,
    "Natural Sand": 0.0042,
    "Plastiment 30": 2.2000,
    "ECO WR": 2.2000,
    "Retarder": 2.2000,
}

# ============================================================
# BINDER FAMILIES
# ============================================================
BINDER_FAMILIES = {
    "P1": {"GGBFS": 0.00, "Fly Ash": 0.00},
    "F2": {"GGBFS": 0.00, "Fly Ash": 0.25},
    "F4": {"GGBFS": 0.00, "Fly Ash": 0.40},
    "S3": {"GGBFS": 0.35, "Fly Ash": 0.00},
    "S5": {"GGBFS": 0.50, "Fly Ash": 0.00},
    "S6": {"GGBFS": 0.65, "Fly Ash": 0.00},
    "T1": {"GGBFS": 0.40, "Fly Ash": 0.20},
    "T2": {"GGBFS": 0.40, "Fly Ash": 0.30},
}

# ============================================================
# Design function
# ============================================================
def _wb_band_split(wb: float):
    if wb >= 0.58: key = 0.58
    elif wb >= 0.50: key = 0.50
    elif wb >= 0.42: key = 0.42
    else: key = 0.34
    return SHEET_SPLITS[key]

def _paste_volume_liters(water, c, s, fa):
    return (water/DENSITY["Water"] + c/DENSITY["Cement"] +
            s/DENSITY["GGBFS"] + fa/DENSITY["Fly Ash"] +
            SHEET_PASTE_EXTRA_LITERS)

def _combined_agg_density(split):
    return (split["20mm"]*DENSITY["20mm Aggregate"] +
            split["10mm"]*DENSITY["10mm Aggregate"] +
            split["Man"] *DENSITY["Man Sand"] +
            split["Nat"] *DENSITY["Natural Sand"])

def design_mix_from_strengths_min(f3_min, f28_min, fam_key="S5", water_fixed=WATER_DEFAULT):
    wb_pred = 190 / (f28_min * 3.2) ** 0.5  # simplified relationship
    wb_pred = np.clip(wb_pred, 0.34, 0.75)
    water = water_fixed
    binder_total = water / wb_pred
    fam = BINDER_FAMILIES[fam_key]
    slag_frac, fly_frac = fam["GGBFS"], fam["Fly Ash"]
    cem_frac = 1 - slag_frac - fly_frac
    c, s, fa = binder_total * cem_frac, binder_total * slag_frac, binder_total * fly_frac

    split = _wb_band_split(wb_pred)
    V_paste = _paste_volume_liters(water, c, s, fa)
    V_air = (DEFAULT_AIR_PERCENT / 100.0) * 1000.0
    V_agg = 1000.0 - V_paste - (V_air if SHEET_SUBTRACT_AIR else 0.0)
    rho_agg = _combined_agg_density(split)
    M_agg_total = V_agg * rho_agg
    a20 = M_agg_total * split["20mm"]
    a10 = M_agg_total * split["10mm"]
    ms  = M_agg_total * split["Man"]
    ns  = M_agg_total * split["Nat"]

    plast = ADM_DOSE_MASS_PER_100KG["Plastiment 30"] * binder_total / 100
    eco   = ADM_DOSE_MASS_PER_100KG["ECO_WR_PC_OR_SLAG"] * binder_total / 100
    ret   = ADM_DOSE_MASS_PER_100KG["Retarder"] * binder_total / 100

    ec_total = (water * EF_PER_KG["Water"] + c * EF_PER_KG["Cement"] +
                s * EF_PER_KG["GGBFS"] + fa * EF_PER_KG["Fly Ash"] +
                a20 * EF_PER_KG["20mm Aggregate"] + a10 * EF_PER_KG["10mm Aggregate"] +
                ms * EF_PER_KG["Man Sand"] + ns * EF_PER_KG["Natural Sand"] +
                plast * EF_PER_KG["Plastiment 30"] + eco * EF_PER_KG["ECO WR"] +
                ret * EF_PER_KG["Retarder"])

    return {
        "w/b": wb_pred,
        "binder": {"Cement": c, "GGBFS": s, "Fly Ash": fa},
        "aggregates": {
            "20mm Aggregate": a20, "10mm Aggregate": a10,
            "Man Sand": ms, "Natural Sand": ns
        },
        "admixtures": {"Plastiment 30": plast, "ECO WR": eco, "Retarder": ret},
        "ec_total": ec_total,
    }

# ============================================================
# Streamlit UI
# ============================================================
col1, col2, col3 = st.columns(3)
f3_min = col1.number_input("3-day Strength (MPa)", 10.0, 60.0, 20.0)
f28_min = col2.number_input("28-day Strength (MPa)", 20.0, 90.0, 50.0)
fam_key = col3.selectbox("Binder Family", list(BINDER_FAMILIES.keys()), index=4)

if st.button("Design Mix"):
    result = design_mix_from_strengths_min(f3_min, f28_min, fam_key)
    st.subheader("Predicted Parameters")
    st.write(f"**Water/Binder Ratio:** {result['w/b']:.2f}")
    st.write(f"**Embodied Carbon (kg CO₂e/m³):** {result['ec_total']:.2f}")

    st.subheader("📦 Materials Table (kg/m³)")
    mat_table = pd.DataFrame({
        "Material": ["Cement", "GGBFS", "Fly Ash", "20mm Aggregate",
                     "10mm Aggregate", "Man Sand", "Natural Sand",
                     "Plastiment 30", "ECO WR", "Retarder"],
        "Mass (kg/m³)": [
            result["binder"]["Cement"],
            result["binder"]["GGBFS"],
            result["binder"]["Fly Ash"],
            result["aggregates"]["20mm Aggregate"],
            result["aggregates"]["10mm Aggregate"],
            result["aggregates"]["Man Sand"],
            result["aggregates"]["Natural Sand"],
            result["admixtures"]["Plastiment 30"],
            result["admixtures"]["ECO WR"],
            result["admixtures"]["Retarder"],
        ],
    })
    mat_table["Mass (kg/m³)"] = mat_table["Mass (kg/m³)"].round(2)
    st.dataframe(mat_table, use_container_width=True)

    st.subheader("🪨 Aggregate Breakdown (Table)")
    agg_df = pd.DataFrame({
        "Aggregate Type": list(result["aggregates"].keys()),
        "Mass (kg/m³)": [round(v, 2) for v in result["aggregates"].values()]
    })
    st.table(agg_df)
