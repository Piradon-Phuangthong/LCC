# app.py
# Strength-Driven Mix Designer — final streamlined version
# Run: pip install streamlit scikit-learn numpy pandas
#      streamlit run app.py

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
st.caption("Fixed-water workflow • Spreadsheet-style aggregates & EC • Curve + KNN w/b prediction")

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

# --- Aggregate splits by w/b band (yellow table) ---
SHEET_SPLITS = {
    0.58: {"20mm": 0.392, "10mm": 0.168, "Nat": 0.264, "Man": 0.176},
    0.50: {"20mm": 0.406, "10mm": 0.174, "Nat": 0.252, "Man": 0.168},
    0.42: {"20mm": 0.420, "10mm": 0.180, "Nat": 0.240, "Man": 0.160},
    0.34: {"20mm": 0.448, "10mm": 0.192, "Nat": 0.216, "Man": 0.144},
}
SHEET_SPLIT_HIGH_WB = SHEET_SPLITS[0.58]

# -------- Admixture doses (per 100 kg binder) --------
ADM_DOSE_MASS_PER_100KG = {
    "Retarder":      0.105,
    "Plastiment 30": 0.315,
    "ECO_WR_FA_OR_TERNARY": 0.214,  # F2, F4, T1, T2
    "ECO_WR_PC_OR_SLAG":    0.535,  # P1, S3, S5, S6
}

# ============================================================
# Dataset and binder families (same as before)
# ============================================================
data2 = {
    "Water/Binder": [0.57,0.57,0.57,0.58,0.57,0.57,0.57,0.55,0.54,
                     0.50,0.50,0.50,0.50,0.50,0.50,0.50,0.44,0.47,
                     0.42,0.41,0.42,0.42,0.42,0.42,0.41,0.40,0.39,
                     0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,0.34,
                     0.66,0.66,0.64,0.66,0.66,0.66,0.66,0.66,0.66],
    "Free Water": [191,192,193,197,191,190,191,182,180,
                   189,192,191,196,191,191,191,173,184,
                   188,189,190,195,188,195,189,184,183,
                   191,192,193,197,193,193,192,193,193,
                   194,195,189,197,194,194,193,194,193],
    "Cement": [338,255,205,171,220,167,118,135,101,
               386,294,234,199,254,195,136,159,119,
               461,346,280,233,297,233,163,190,141,
               574,432,348,290,372,286,199,228,171,
               295,222,177,149,191,147,103,117,88],
    "GGBFS": [0,0,0,0,118,167,218,135,135,
              0,0,0,0,137,195,253,159,159,
              0,0,0,0,161,233,304,190,187,
              0,0,0,0,200,286,369,228,228,
              0,0,0,0,103,147,190,117,118],
    "Fly Ash": [0,84,137,171,0,0,0,68,101,
                0,98,156,199,0,0,0,79,119,
                0,117,186,233,0,0,0,95,141,
                0,144,232,290,0,0,0,114,171,
                0,74,118,149,0,0,0,59,88],
    "Strength3d": [31.5,24.3,20.5,16.2,19.0,12.5,11.2,13.4,11.9,
                   36.2,30.6,27.2,22.3,24.0,18.7,15.9,20.7,15.2,
                   48.4,40.4,35.6,31.3,28.9,23.0,17.0,24.9,20.7,
                   55.3,45.0,45.3,44.5,37.8,32.8,23.0,28.6,25.3,
                   21.5,16.2,13.6,11.6,12.5,9.2,7.9,9.3,7.8],
    "Strength28d": [49.6,45.6,36.7,34.0,43.7,42.4,37.0,44.3,37.6,
                    54.1,54.0,48.8,44.0,55.3,47.4,41.4,51.9,47.1,
                    70.7,66.5,60.0,58.5,62.7,58.8,47.4,58.6,55.6,
                    83.3,78.6,74.5,72.9,72.7,67.3,56.4,61.9,61.8,
                    34.6,30.6,28.9,24.3,36.8,35.3,29.8,31.3,28.1]
}
df = pd.DataFrame(data2)

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
# Fitting models (unchanged logic)
# ============================================================
def _power_model(x, A, b):
    return A * (x ** b)

def fit_abram_curve(df_sub):
    x = df_sub["Water/Binder"].values.astype(float)
    y = df_sub["Strength28d"].values.astype(float)
    m = (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 2:
        return 100.0, -1.8
    try:
        from scipy.optimize import curve_fit
        b0, a0 = np.polyfit(np.log(x), np.log(y), 1)
        A0 = float(np.exp(a0))
        (A, b), _ = curve_fit(_power_model, x, y, p0=(A0, b0))
        return float(A), float(b)
    except Exception:
        lx, ly = np.log(x), np.log(y)
        b, a = np.polyfit(lx, ly, 1)
        A = float(np.exp(a))
        return float(A), float(b)

family_curves = {}
for fam in BINDER_FAMILIES.keys():
    sub = df.copy()
    family_curves[fam] = fit_abram_curve(sub)
A_g, b_g = fit_abram_curve(df)

def predict_wb_from_f28_curve(f28, fam_key):
    A, b = family_curves.get(fam_key, (A_g, b_g))
    return max(0.34, min(0.75, (max(1e-6, f28) / A) ** (1.0 / b)))

# ============================================================
# Sheet math helpers
# ============================================================
def _wb_band_split(wb: float):
    if wb >= 0.58: key = 0.58
    elif wb >= 0.50: key = 0.50
    elif wb >= 0.42: key = 0.42
    else: key = 0.34
    return SHEET_SPLITS[key]

def _paste_volume_liters(water_kg, c_kg, s_kg, fa_kg):
    return (water_kg/DENSITY["Water"]
            + c_kg/DENSITY["Cement"]
            + s_kg/DENSITY["GGBFS"]
            + fa_kg/DENSITY["Fly Ash"]
            + SHEET_PASTE_EXTRA_LITERS)

def _combined_agg_density_kg_per_L(split):
    return (split["20mm"]*DENSITY["20mm Aggregate"]
            + split["10mm"]*DENSITY["10mm Aggregate"]
            + split["Man"] *DENSITY["Man Sand"]
            + split["Nat"] *DENSITY["Natural Sand"])

# ============================================================
# Main design function
# ============================================================
def design_mix(f3_min, f28_min, fam_key, water_fixed):
    wb = predict_wb_from_f28_curve(f28_min, fam_key)
    binder_total = water_fixed / wb
    frac = BINDER_FAMILIES[fam_key]
    slag, fly = frac["GGBFS"], frac["Fly Ash"]
    cem = 1 - slag - fly
    c, s, fa = binder_total*cem, binder_total*slag, binder_total*fly

    plast = ADM_DOSE_MASS_PER_100KG["Plastiment 30"] * binder_total / 100
    eco = (ADM_DOSE_MASS_PER_100KG["ECO_WR_FA_OR_TERNARY"]
           if fly > 0 else ADM_DOSE_MASS_PER_100KG["ECO_WR_PC_OR_SLAG"]) * binder_total / 100
    ret = ADM_DOSE_MASS_PER_100KG["Retarder"] * binder_total / 100
    adm_total = plast + eco + ret

    split = _wb_band_split(wb)
    V_paste = _paste_volume_liters(water_fixed, c, s, fa)
    V_air = (DEFAULT_AIR_PERCENT/100)*1000
    V_agg = 1000 - V_paste - (V_air if SHEET_SUBTRACT_AIR else 0)
    rho_agg = _combined_agg_density_kg_per_L(split)
    M_agg = V_agg * rho_agg
    a20, a10, ms, ns = [M_agg*split[k] for k in ["20mm","10mm","Man","Nat"]]

    total_mass = water_fixed + binder_total + adm_total + M_agg
    return {
        "wb": wb, "binder_total": binder_total, "rho": total_mass,
        "binder": {"Cement": c, "GGBFS": s, "Fly Ash": fa},
        "agg": {"20mm Aggregate": a20, "10mm Aggregate": a10, "Man Sand": ms, "Natural Sand": ns},
        "adm": {"Plastiment 30": plast, "ECO WR": eco, "Retarder": ret}
    }

# ============================================================
# Sidebar inputs
# ============================================================
with st.sidebar:
    st.header("Inputs")
    water = st.number_input("Fixed water (kg/m³)", 120.0, 240.0, WATER_DEFAULT, 1.0)
    fam = st.selectbox("Binder family", list(BINDER_FAMILIES.keys()), index=4)
    col1, col2 = st.columns(2)
    with col1:
        f3 = st.number_input("3-day strength (MPa)", 5.0, 80.0, 30.0, 0.5)
    with col2:
        f28 = st.number_input("28-day strength (MPa)", 10.0, 100.0, 50.0, 0.5)
    run_btn = st.button("Design mix", type="primary", use_container_width=True)

# ============================================================
# Binder family panel
# ============================================================
with st.expander("Binder family mixes (percent of total binder)", expanded=True):
    df_fam = pd.DataFrame(
        [(k, 100*(1-v["GGBFS"]-v["Fly Ash"]), 100*v["GGBFS"], 100*v["Fly Ash"])
         for k,v in BINDER_FAMILIES.items()],
        columns=["Family","Cement %","GGBFS %","Fly Ash %"]
    )
    st.dataframe(df_fam, use_container_width=True, hide_index=True)

# ============================================================
# Main Output
# ============================================================
if run_btn:
    out = design_mix(f3, f28, fam, water)

    colA, colB, colC = st.columns(3)
    colA.metric("w/b", f"{out['wb']:.3f}")
    colB.metric("Binder total (kg/m³)", f"{out['binder_total']:.2f}")
    colC.metric("Fresh density (kg/m³)", f"{out['rho']:.2f}")

    # Materials table (2 decimals)
    st.subheader("Materials Table")
    mats = []
    mats.append(["Water", DENSITY["Water"], water, water/DENSITY["Water"]])
    for k,v in out["binder"].items():
        mats.append([k, DENSITY[k], v, v/DENSITY[k]])
    for k,v in out["agg"].items():
        mats.append([k, DENSITY[k], v, v/DENSITY[k]])
    for k,v in out["adm"].items():
        mats.append([k, DENSITY[k], v, v/DENSITY[k]])

    df_mat = pd.DataFrame(mats, columns=["Material","Density (kg/L)","Mass (kg)","Volume (L)"])
    df_mat["Mass (kg)"] = df_mat["Mass (kg)"].map(lambda x: f"{x:.2f}")
    df_mat["Volume (L)"] = df_mat["Volume (L)"].map(lambda x: f"{x:.2f}")
    st.dataframe(df_mat, use_container_width=True, hide_index=True)

    # Aggregates table (instead of graph)
    st.subheader("Aggregates (kg/m³)")
    df_ag = pd.DataFrame(list(out["agg"].items()), columns=["Aggregate","kg/m³"])
    df_ag["kg/m³"] = df_ag["kg/m³"].map(lambda x: f"{x:.2f}")
    st.dataframe(df_ag, use_container_width=True, hide_index=True)
