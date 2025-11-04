# app.py
# Streamlit UI for Strength-Driven Low-Carbon Concrete Mix Designer
# ---------------------------------------------------------------
# How to run:
#   pip install streamlit scikit-learn numpy pandas
#   (scipy is optional; the code falls back if missing)
#   streamlit run app.py

import math
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsRegressor

# ===========================
# Streamlit page config
# ===========================
st.set_page_config(
    page_title="Strength-Driven Mix Designer",
    page_icon="🧱",
    layout="wide"
)

st.title("🧱 Strength-Driven Mix Designer (Curve + KNN • Spreadsheet EC)")
st.caption("Your core calculator, wrapped in a simple Streamlit UI.")

# ------------------------------------------------------------
# Defaults (can be overridden via sidebar controls)
# ------------------------------------------------------------
USE_FIXED_WATER = True     # True = validation mode; False = design mode (predict water)
WATER_FIXED = 190.0        # kg/m³ when validation is ON

SHEET_METHOD = True
SHEET_PASTE_EXTRA_LITERS = 15.0
SHEET_SUBTRACT_AIR = False

# ============================
# Material densities (kg/L)
# ============================
DENSITY = {
    "Water": 1.00,
    "Cement": 3.11,        # PC
    "Fly Ash": 2.35,
    "GGBFS": 2.89,         # Slag
    "20mm Aggregate": 2.61,
    "10mm Aggregate": 2.71,
    "Man Sand": 2.64,      # manufactured sand
    "Natural Sand": 2.65,
    "Plastiment 30": 1.05,
    "ECO WR": 1.07,
    "Retarder": 1.05,
}

# From your yellow table (rates). Mapping:
#  - "Fine Sand"  -> "Natural Sand"
#  - "Coarse"     -> "Manufactured Sand"
SHEET_SPLITS = {
    0.58: {"20mm": 0.392, "10mm": 0.168, "Nat": 0.264, "Man": 0.176},
    0.50: {"20mm": 0.406, "10mm": 0.174, "Nat": 0.252, "Man": 0.168},
    0.42: {"20mm": 0.420, "10mm": 0.180, "Nat": 0.240, "Man": 0.160},
    0.34: {"20mm": 0.448, "10mm": 0.192, "Nat": 0.216, "Man": 0.144},
}
SHEET_SPLIT_HIGH_WB = SHEET_SPLITS[0.58]

# -------- Admixture doses from your sheet (per 100 kg binder) --------
ADM_DOSE_MASS_PER_100KG = {
    "Retarder":      0.105,  # All binders
    "Plastiment 30": 0.315,  # All binders (low-range WR)
    # ECO WR depends on binder type:
    "ECO_WR_FA_OR_TERNARY": 0.214,  # FA & ternary (F2, F4, T1, T2)
    "ECO_WR_PC_OR_SLAG":    0.535,  # GP (P1) & Slag blends (S3, S5, S6)
}

def _family_uses_flyash(fam_key: str) -> bool:
    fam = BINDER_FAMILIES.get(fam_key.upper(), {})
    return (fam.get("Fly Ash", 0.0) or 0.0) > 0.0

def compute_admixtures_from_sheet(binder_total_kg_m3: float, fam_key: str):
    scale = binder_total_kg_m3 / 100.0
    ret_kg   = ADM_DOSE_MASS_PER_100KG["Retarder"]      * scale
    plast_kg = ADM_DOSE_MASS_PER_100KG["Plastiment 30"] * scale
    eco_key  = "ECO_WR_FA_OR_TERNARY" if _family_uses_flyash(fam_key) else "ECO_WR_PC_OR_SLAG"
    eco_kg   = ADM_DOSE_MASS_PER_100KG[eco_key] * scale
    return plast_kg, eco_kg, ret_kg

def _wb_band_split(wb: float):
    if wb >= 0.58: key = 0.58
    elif wb >= 0.50: key = 0.50
    elif wb >= 0.42: key = 0.42
    else: key = 0.34
    return SHEET_SPLITS[key]

def _paste_volume_liters(water_kg, c_kg, s_kg, fa_kg, densities=DENSITY):
    return (water_kg/densities["Water"]
            + c_kg/densities["Cement"]
            + s_kg/densities["GGBFS"]
            + fa_kg/densities["Fly Ash"]
            + SHEET_PASTE_EXTRA_LITERS)

def _combined_agg_density_kg_per_L(split, densities=DENSITY):
    return (split["20mm"]*densities["20mm Aggregate"]
            + split["10mm"]*densities["10mm Aggregate"]
            + split["Man"] *densities["Man Sand"]
            + split["Nat"] *densities["Natural Sand"])

# ============================================================
# Dataset (45 mixes)
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
ec_list = [358,280,235,206,275,233,199,197,168,
           403,318,268,233,310,269,222,226,190,
           470,366,307,267,366,306,258,261,218,
           581,453,375,327,431,373,306,306,255,
           299,227,188,163,223,192,159,155,129]
df = pd.DataFrame(data2)
df["EC_exp"] = ec_list

# ============================================================
# Binder families (mass fractions of binder)
# ============================================================
BINDER_FAMILIES = {
    "P1": {"GGBFS": 0.00, "Fly Ash": 0.00},  # 100% PC
    "F2": {"GGBFS": 0.00, "Fly Ash": 0.25},  # 25% Fly
    "F4": {"GGBFS": 0.00, "Fly Ash": 0.40},  # 40% Fly
    "S3": {"GGBFS": 0.35, "Fly Ash": 0.00},  # 35% Slag
    "S5": {"GGBFS": 0.50, "Fly Ash": 0.00},  # 50% Slag
    "S6": {"GGBFS": 0.65, "Fly Ash": 0.00},  # 65% Slag
    "T1": {"GGBFS": 0.40, "Fly Ash": 0.20},  # 40% Slag + 20% Fly
    "T2": {"GGBFS": 0.40, "Fly Ash": 0.30},  # 40% Slag + 30% Fly
}
FAMILY_PROTOTYPES = {k: (v["GGBFS"], v["Fly Ash"]) for k, v in BINDER_FAMILIES.items()}

def _closest_family(slag_frac, fly_frac):
    best_key, best_dist = None, 1e9
    for k, (s, f) in FAMILY_PROTOTYPES.items():
        d = (slag_frac - s)**2 + (fly_frac - f)**2
        if d < best_dist:
            best_key, best_dist = k, d
    return best_key

# Label dataset rows by nearest (slag%, fly%)
binder_sum = (df["Cement"] + df["GGBFS"] + df["Fly Ash"]).replace(0, 1e-9)
df["_slag_frac"] = df["GGBFS"] / binder_sum
df["_fly_frac"]  = df["Fly Ash"] / binder_sum
df["_family"] = [_closest_family(s, f) for s, f in zip(df["_slag_frac"], df["_fly_frac"])]

# ============================================================
# Abram-style curves per family: fit in LINEAR space
# ============================================================
def _power_model(x, A, b):
    return A * (x ** b)

def _r2_linear(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

def fit_abram_curve(df_sub):
    """
    Fit f28 = A * (w/b)^b in LINEAR space (minimizes absolute strength error).
    Falls back to log–log fit if SciPy isn't available.
    """
    x = df_sub["Water/Binder"].values.astype(float)
    y = df_sub["Strength28d"].values.astype(float)
    m = (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 2:
        return 100.0, -1.8
    try:
        from scipy.optimize import curve_fit  # optional
        b0, a0 = np.polyfit(np.log(x), np.log(y), 1)
        A0 = float(np.exp(a0))
        (A, b), _ = curve_fit(
            _power_model, x, y,
            p0=(A0, b0),
            bounds=([1e-6, -4.0], [1e3, -0.6])
        )
        return float(A), float(b)
    except Exception:
        lx, ly = np.log(x), np.log(y)
        b, a = np.polyfit(lx, ly, 1)
        A = float(np.exp(a))
        if b < -2.1:  # avoid over-curvature from tiny samples
            b = -1.9
        return float(A), float(b)

# === Family-level inverse models: Strength28d -> Water/Binder ===
wb_inv_models = {}
wb_ranges = {}
for fam in BINDER_FAMILIES.keys():
    sub = df[df["_family"] == fam][["Strength28d", "Water/Binder"]].dropna()
    if len(sub) >= 2:
        mdl = KNeighborsRegressor(n_neighbors=min(4, len(sub)), weights="distance")
        mdl.fit(sub[["Strength28d"]].values, sub["Water/Binder"].values)
        wb_inv_models[fam] = mdl
        wb_ranges[fam] = (float(sub["Strength28d"].min()), float(sub["Strength28d"].max()))

# === Build family-specific curves and global fallback ===
family_curves = {}
for fam in BINDER_FAMILIES.keys():
    sub = df[df["_family"] == fam]
    if len(sub) >= 2:
        family_curves[fam] = fit_abram_curve(sub)
A_g, b_g = fit_abram_curve(df)

# === 3-day strength models (reporting only) ===
family_models_f3 = {}
for fam in BINDER_FAMILIES.keys():
    sub = df[df["_family"] == fam][["Water/Binder", "Strength3d"]].dropna()
    if len(sub) >= 2:
        family_models_f3[fam] = KNeighborsRegressor(
            n_neighbors=min(4, len(sub)), weights="distance"
        ).fit(sub[["Water/Binder"]].values, sub["Strength3d"].values)

# global fallback uses wb + blend fractions
knn_f3_global = KNeighborsRegressor(n_neighbors=4, weights="distance").fit(
    df[["Water/Binder", "_slag_frac", "_fly_frac"]].values, df["Strength3d"].values
)

def predict_f3_from_wb(wb: float, fam_key: str) -> float:
    mdl = family_models_f3.get(fam_key)
    if mdl is not None:
        return float(mdl.predict([[wb]])[0])
    fam = BINDER_FAMILIES.get(fam_key, {})
    s = float(fam.get("GGBFS", 0.0)); f = float(fam.get("Fly Ash", 0.0))
    return float(knn_f3_global.predict([[wb, s, f]])[0])

def recommend_wb_for_f3(target_f3: float, fam_key: str, wb_start: float) -> float:
    cur = predict_f3_from_wb(wb_start, fam_key)
    if cur >= target_f3:
        return round(wb_start, 3)
    lo, hi = 0.34, max(0.34, min(0.75, wb_start))
    if predict_f3_from_wb(lo, fam_key) < target_f3:
        return round(lo, 3)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if predict_f3_from_wb(mid, fam_key) >= target_f3:
            lo = mid
        else:
            hi = mid
    return round(hi, 3)

def _wb_from_curve(f28, fam_key):
    A, b = family_curves.get(fam_key, (A_g, b_g))
    if abs(b) < 1e-9:
        return 0.50
    return float((max(1e-6, f28) / A) ** (1.0 / b))

def _wb_from_knn(f28, fam_key):
    mdl = wb_inv_models.get(fam_key)
    if mdl is None:
        return None
    fmin, fmax = wb_ranges[fam_key]
    f_in = min(max(float(f28), fmin), fmax)
    return float(mdl.predict([[f_in]])[0])

def predict_wb_from_f28_curve(f28, fam_key):
    wb_curve = _wb_from_curve(f28, fam_key)
    wb_knn = _wb_from_knn(f28, fam_key)
    if wb_knn is None:
        wb = wb_curve
    else:
        f = float(f28)
        w_data = 0.7 if f <= 40 else (0.4 if f >= 60 else 0.7 - (0.3*(f-40)/20.0))
        wb = w_data*wb_knn + (1.0 - w_data)*wb_curve
    return max(0.34, min(0.75, wb))

def implied_f28_from_wb(wb, fam_key):
    A, b = family_curves.get(fam_key, (A_g, b_g))
    return float(A * (wb ** b))

# ============================================================
# Water prediction model (used only in design mode)
# ============================================================
family_models_water = {}
for fam in BINDER_FAMILIES.keys():
    sub = df[df["_family"] == fam]
    if len(sub) >= 2:
        family_models_water[fam] = KNeighborsRegressor(n_neighbors=3, weights="distance") \
            .fit(sub[["Strength3d","Strength28d","Water/Binder"]].values, sub["Free Water"].values)

knn_water_global = KNeighborsRegressor(n_neighbors=3, weights="distance") \
    .fit(df[["Strength3d","Strength28d","Water/Binder"]].values, df["Free Water"].values)

W_MIN, W_MAX = float(df["Free Water"].min()), float(df["Free Water"].max())

def get_water(f3, f28, wb, fam_key):
    if st.session_state.get("use_fixed_water", USE_FIXED_WATER):
        return st.session_state.get("water_fixed", WATER_FIXED)
    mdl_w = family_models_water.get(fam_key, knn_water_global)
    w = float(mdl_w.predict([[f3, f28, wb]])[0])
    return max(W_MIN, min(W_MAX, w))

# ============================================================
# Aggregate templates by w/b family (fallback path retained)
# ============================================================
wb_families = [
    (0.57, (726, 318, 333, 497), 2407, 1.8),
    (0.50, (720, 315, 315, 462), 2390, 1.8),
    (0.42, (718, 314, 294, 441), 2420, 1.9),
    (0.34, (727, 318, 252, 374), 2442, 1.4),
    (0.66, (721, 317, 346, 516), 2392, 1.9),
]
def nearest_family(wb):
    return min(wb_families, key=lambda t: abs(t[0] - wb))

def _normalized_split_from_template(center_wb):
    for cw, (a20, a10, ms, ns), _, _ in wb_families:
        if abs(cw - center_wb) < 1e-6:
            s = a20 + a10 + ms + ns
            return {"20mm": a20/s, "10mm": a10/s, "Man": ms/s, "Nat": ns/s}
    return SHEET_SPLIT_HIGH_WB

def _split_for_wb_band(wb):
    if wb >= 0.58:
        return SHEET_SPLIT_HIGH_WB
    elif 0.50 <= wb < 0.58:
        return _normalized_split_from_template(0.50)
    elif 0.42 <= wb < 0.50:
        return _normalized_split_from_template(0.42)
    else:
        return _normalized_split_from_template(0.34)

# ============================================================
# Spreadsheet EC factors (kg CO2e per kg material)
# ============================================================
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

def compute_ec_from_mix_spreadsheet(water, binder_split, aggs, admix_tuple):
    ec = 0.0
    ec += water * EF_PER_KG["Water"]
    ec += binder_split["Cement"] * EF_PER_KG["Cement"]
    ec += binder_split["GGBFS"]  * EF_PER_KG["GGBFS"]
    ec += binder_split["Fly Ash"] * EF_PER_KG["Fly Ash"]
    ec += aggs["20mm Aggregate"] * EF_PER_KG["20mm Aggregate"]
    ec += aggs["10mm Aggregate"] * EF_PER_KG["10mm Aggregate"]
    ec += aggs["Man Sand"]       * EF_PER_KG["Man Sand"]
    ec += aggs["Natural Sand"]   * EF_PER_KG["Natural Sand"]
    plast, eco, ret = admix_tuple
    ec += plast * EF_PER_KG["Plastiment 30"]
    ec += eco   * EF_PER_KG["ECO WR"]
    ec += ret   * EF_PER_KG["Retarder"]
    return ec

# ============================================================
# Core design function
# ============================================================
def design_mix_from_strengths_min(f3_min, f28_min, binder_family_key="S5"):
    fam_key = binder_family_key.upper()

    # 1) w/b from f28 (blended inverse)
    wb_pred = predict_wb_from_f28_curve(f28_min, fam_key)

    # 2) Water (fixed or predicted)
    water_pred = get_water(f3_min, f28_min, wb_pred, fam_key)

    # 3) Binder total + split per family
    binder_total = water_pred / wb_pred
    fam = BINDER_FAMILIES[fam_key]
    slag_frac, fly_frac = fam.get("GGBFS", 0.0), fam.get("Fly Ash", 0.0)
    cem_frac = max(0.0, 1.0 - slag_frac - fly_frac)
    c, s, fa = binder_total*cem_frac, binder_total*slag_frac, binder_total*fly_frac

    # 4) Admixtures — per 100 kg binder (sheet-accurate)
    plast, eco, ret = compute_admixtures_from_sheet(binder_total, fam_key)
    adm_total = plast + eco + ret

    # 5) Aggregates from sheet logic (or template fallback)
    if st.session_state.get("sheet_method", SHEET_METHOD):
        split = _wb_band_split(wb_pred)
        V_paste_L = _paste_volume_liters(water_pred, c, s, fa)
        air_pct = st.session_state.get("air_percent", 1.9)
        V_air_L = (air_pct/100.0) * 1000.0
        V_agg_L = max(0.0, 1000.0 - V_paste_L - (V_air_L if st.session_state.get("sheet_subtract_air", SHEET_SUBTRACT_AIR) else 0.0))
        rho_agg_kg_per_L = _combined_agg_density_kg_per_L(split)

        M_agg_total = V_agg_L * rho_agg_kg_per_L
        a20 = M_agg_total * split["20mm"]
        a10 = M_agg_total * split["10mm"]
        ms  = M_agg_total * split["Man"]
        ns  = M_agg_total * split["Nat"]

        rho_target = water_pred + binder_total + adm_total + M_agg_total
        center_wb = wb_pred
    else:
        center_wb, (a20_b, a10_b, ms_b, ns_b), rho_target, air_pct = nearest_family(wb_pred)
        base_ag = a20_b + a10_b + ms_b + ns_b
        non_ag = water_pred + binder_total + adm_total
        aggs_needed = max(0.0, rho_target - non_ag)
        scale = aggs_needed/base_ag if base_ag > 0 else 1.0
        a20, a10, ms, ns = a20_b*scale, a10_b*scale, ms_b*scale, ns_b*scale

    # 6) EC
    ec_calc = compute_ec_from_mix_spreadsheet(
        water_pred,
        {"Cement": c, "GGBFS": s, "Fly Ash": fa},
        {"20mm Aggregate": a20, "10mm Aggregate": a10, "Man Sand": ms, "Natural Sand": ns},
        (plast, eco, ret)
    )

    total_mass = water_pred + binder_total + (a20 + a10 + ms + ns) + adm_total

    return {
        "inputs": {"min_3d_MPa": f3_min, "min_28d_MPa": f28_min, "binder_family": fam_key},
        "predicted_parameters": {
            "water_binder_ratio": wb_pred,
            "water_kg_m3": water_pred,
            "binder_total_kg_m3": binder_total,
            "fresh_density_target_kg_m3": rho_target,  # "Sum Concrete"
            "air_percent": air_pct,
            "wb_family_center": center_wb,
        },
        "binder_exact": {"Cement": c, "GGBFS": s, "Fly Ash": fa},
        "admixture_split_kg_m3": {"Plastiment 30": plast, "ECO WR": eco, "Retarder": ret},
        "aggregates_exact": {"20mm Aggregate": a20, "10mm Aggregate": a10, "Man Sand": ms, "Natural Sand": ns},
        "embodied_carbon": {"calculated_from_EF_spreadsheet": ec_calc},
        "totals": {"sum_all_components_kg_m3": total_mass}
    }

# ============================================================
# Sidebar controls
# ============================================================
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio(
        "Mode",
        options=["Validation (fixed water)", "Design (predict water)"],
        index=0 if USE_FIXED_WATER else 1
    )
    st.session_state["use_fixed_water"] = (mode == "Validation (fixed water)")

    if st.session_state["use_fixed_water"]:
        st.session_state["water_fixed"] = st.number_input("Fixed water (kg/m³)", 120.0, 240.0, WATER_FIXED, 1.0)

    st.divider()
    st.session_state["sheet_method"] = st.toggle("Use spreadsheet aggregate method", value=SHEET_METHOD)
    st.session_state["sheet_subtract_air"] = st.toggle("Subtract air before sizing aggregates", value=SHEET_SUBTRACT_AIR)
    st.session_state["paste_extra_liters"] = st.number_input("Paste + extra allowance (L)", 0.0, 40.0, SHEET_PASTE_EXTRA_LITERS, 1.0)
    SHEET_PASTE_EXTRA_LITERS = st.session_state["paste_extra_liters"]  # bind to global used in calc

    st.session_state["air_percent"] = st.number_input("Air content (%)", 0.0, 6.0, 1.9, 0.1)

    st.divider()
    fam_keys = list(BINDER_FAMILIES.keys())
    default_idx = fam_keys.index("S5") if "S5" in fam_keys else 0
    fam = st.selectbox("Binder family", fam_keys, index=default_idx)

    colA, colB = st.columns(2)
    with colA:
        f3_min = st.number_input("Minimum 3-day strength (MPa)", 5.0, 80.0, 30.0, 0.5)
    with colB:
        f28_min = st.number_input("Minimum 28-day strength (MPa)", 10.0, 100.0, 50.0, 0.5)

    run_btn = st.button("Design mix", type="primary", use_container_width=True)

# ============================================================
# Main UI
# ============================================================
def material_table(out: dict) -> pd.DataFrame:
    w = out["predicted_parameters"]["water_kg_m3"]
    b = out["binder_exact"]
    a = out["aggregates_exact"]
    adm = out["admixture_split_kg_m3"]

    rows = [
        ("Water", DENSITY["Water"], w),
        ("Cement", DENSITY["Cement"], b["Cement"]),
        ("Fly Ash", DENSITY["Fly Ash"], b["Fly Ash"]),
        ("GGBFS", DENSITY["GGBFS"], b["GGBFS"]),
        ("20mm Aggregate", DENSITY["20mm Aggregate"], a["20mm Aggregate"]),
        ("10mm Aggregate", DENSITY["10mm Aggregate"], a["10mm Aggregate"]),
        ("Man Sand", DENSITY["Man Sand"], a["Man Sand"]),
        ("Natural Sand", DENSITY["Natural Sand"], a["Natural Sand"]),
        ("Plastiment 30", DENSITY["Plastiment 30"], adm["Plastiment 30"]),
        ("ECO WR", DENSITY["ECO WR"], adm["ECO WR"]),
        ("Retarder", DENSITY["Retarder"], adm["Retarder"]),
    ]
    df_rows = []
    total_mass = 0.0
    total_vol = 0.0
    for name, dens, mass in rows:
        vol = mass/dens if dens > 0 else 0.0
        df_rows.append([name, dens, mass, vol])
        total_mass += mass
        total_vol += vol

    # Add Air row
    air_vol_L = (out["predicted_parameters"]["air_percent"]/100.0)*1000.0
    df_rows.append(["Air", np.nan, 0.0, air_vol_L])
    total_vol += air_vol_L

    df_out = pd.DataFrame(df_rows, columns=["Material", "Density (kg/L)", "Mass (kg)", "Volume (L)"])
    df_out["Mass %"] = 100.0 * df_out["Mass (kg)"] / total_mass
    avg_density = total_mass / total_vol if total_vol else 0.0

    # Append a total row (for display only)
    df_total = pd.DataFrame([["Total (avg density)", round(avg_density,2), round(total_mass,2), round(total_vol,2), ""]], columns=df_out.columns)
    return pd.concat([df_out, df_total], ignore_index=True)

if run_btn:
    out = design_mix_from_strengths_min(f3_min, f28_min, fam)

    pp = out["predicted_parameters"]
    wb   = pp["water_binder_ratio"]
    water= pp["water_kg_m3"]
    btot = pp["binder_total_kg_m3"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Binder family", out["inputs"]["binder_family"])
    m2.metric("w/b (curve+data)", f"{wb:.3f}")
    m3.metric("Water (kg/m³)", f"{water:.1f}")
    m4.metric("Binder total (kg/m³)", f"{btot:.1f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sum Concrete (kg/m³)", f"{pp['fresh_density_target_kg_m3']:.1f}")
    c2.metric("Air (%)", f"{pp['air_percent']:.1f}")
    c3.metric("EC (kg CO₂e/m³)", f"{out['embodied_carbon']['calculated_from_EF_spreadsheet']:.1f}")

    # 3-day reporting + recommendation
    f3_pred_report = predict_f3_from_wb(wb, fam)
    st.subheader("3-Day Strength Check (reporting only)")
    if f3_pred_report + 1e-9 < f3_min:
        wb_rec = recommend_wb_for_f3(f3_min, fam, wb)
        f3_at_rec  = predict_f3_from_wb(wb_rec, fam)
        f28_at_rec = implied_f28_from_wb(wb_rec, fam)
        st.warning(
            f"Predicted f3 ≈ **{f3_pred_report:.1f} MPa** < target **{f3_min:.1f} MPa**.\n\n"
            f"**Recommended w/b:** {wb_rec:.3f} → f3 ≈ {f3_at_rec:.1f} MPa, f28 ≈ {f28_at_rec:.1f} MPa "
            "(design remains unchanged; this is a suggestion)."
        )
    else:
        st.success(f"Predicted f3 ≈ **{f3_pred_report:.1f} MPa** ≥ target **{f3_min:.1f} MPa** — OK at current w/b.")

    st.divider()
    st.subheader("Detailed Materials Table")
    df_mat = material_table(out)
    st.dataframe(df_mat, use_container_width=True, hide_index=True)

    # Aggregates chart
    st.subheader("Aggregates (kg/m³)")
    aggs_df = pd.DataFrame({
        "Aggregate": list(out["aggregates_exact"].keys()),
        "kg/m³": list(out["aggregates_exact"].values())
    })
    st.bar_chart(aggs_df.set_index("Aggregate"))

    # Binder split display
    st.subheader("Binder Split")
    b = out["binder_exact"]
    binder_split_df = pd.DataFrame({
        "Component": ["Cement", "GGBFS", "Fly Ash"],
        "kg/m³": [b["Cement"], b["GGBFS"], b["Fly Ash"]]
    })
    binder_split_df["% of binder"] = 100*binder_split_df["kg/m³"]/btot
    st.dataframe(binder_split_df, use_container_width=True, hide_index=True)

    # Downloads
    st.divider()
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download materials table (CSV)",
            data=df_mat.to_csv(index=False),
            file_name="mix_materials_table.csv",
            mime="text/csv",
            use_container_width=True
        )
    with col_dl2:
        st.download_button(
            "⬇️ Download full result (JSON)",
            data=json.dumps(out, indent=2),
            file_name="mix_design_result.json",
            mime="application/json",
            use_container_width=True
        )

# Helpful reference panel
with st.expander("What does this tool do?"):
    st.markdown("""
- **Inputs:** target strengths (3-day, 28-day) and **binder family** (P1, F2, F4, S3, S5, S6, T1, T2).
- **w/b prediction:** blends a fitted Abram-type curve with KNN inverse models (leans on data at low strengths).
- **Water:** fixed (Validation) or data-driven (Design).
- **Aggregates:** spreadsheet-identical volumetric closure (with +15 L paste allowance, configurable), or template fallback.
- **Admixtures:** per-100-kg binder doses (sheet-accurate).
- **EC:** computed with your NSW-specific factors.
- **3-day check:** reports predicted f3 and suggests a minimal w/b tightening if needed (no auto changes).
""")
