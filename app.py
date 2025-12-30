# app.py
# Strength-Driven Mix Designer — Streamlit UI
# Run:
#   pip install streamlit scikit-learn numpy pandas scipy
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
st.caption("Spreadsheet-style aggregates • Curve+KNN w/b prediction • Report-aligned EC (A1+A2+A3)")

# ===========================
# Core constants
# ===========================
WATER_DEFAULT = 190.0               # kg/m³ (fixed-water mode default)
SHEET_PASTE_EXTRA_LITERS = 15.0     # +15 L allowance identical to your sheet
SHEET_SUBTRACT_AIR = False          # Excel uses Vagg = 1000 - Vpaste (no air subtraction); keep False
DEFAULT_AIR_PERCENT = 1.9           # %

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

# --- Aggregate splits by w/b band (your yellow table) ---
SHEET_SPLITS = {
    0.58: {"20mm": 0.392, "10mm": 0.168, "Nat": 0.264, "Man": 0.176},
    0.50: {"20mm": 0.406, "10mm": 0.174, "Nat": 0.252, "Man": 0.168},
    0.42: {"20mm": 0.420, "10mm": 0.180, "Nat": 0.240, "Man": 0.160},
    0.34: {"20mm": 0.448, "10mm": 0.192, "Nat": 0.216, "Man": 0.144},
}

# -------- Admixture doses (per 100 kg binder) --------
ADM_DOSE_MASS_PER_100KG = {
    "Retarder":      0.105,
    "Plastiment 30": 0.315,
    "ECO_WR_FA_OR_TERNARY": 0.214,  # F2, F4, T1, T2
    "ECO_WR_PC_OR_SLAG":    0.535,  # P1, S3, S5, S6
}

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
# Keep original EC_exp list (for reference only; design uses report-aligned EC)
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
    "P1": {"GGBFS": 0.00, "Fly Ash": 0.00},
    "F2": {"GGBFS": 0.00, "Fly Ash": 0.25},
    "F4": {"GGBFS": 0.00, "Fly Ash": 0.40},
    "S3": {"GGBFS": 0.35, "Fly Ash": 0.00},
    "S5": {"GGBFS": 0.50, "Fly Ash": 0.00},
    "S6": {"GGBFS": 0.65, "Fly Ash": 0.00},
    "T1": {"GGBFS": 0.40, "Fly Ash": 0.20},
    "T2": {"GGBFS": 0.40, "Fly Ash": 0.30},
}
FAMILY_PROTOTYPES = {k: (v["GGBFS"], v["Fly Ash"]) for k, v in BINDER_FAMILIES.items()}

def _closest_family(slag_frac, fly_frac):
    best_key, best_dist = None, 1e9
    for k, (s, f) in FAMILY_PROTOTYPES.items():
        d = (slag_frac - s) ** 2 + (fly_frac - f) ** 2
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
        if b < -2.1:
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

# === 3-day strength models (reporting only; used for optional enforcement) ===
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
    s = float(fam.get("GGBFS", 0.0))
    f = float(fam.get("Fly Ash", 0.0))
    return float(knn_f3_global.predict([[wb, s, f]])[0])

def recommend_wb_for_f3(target_f3: float, fam_key: str, wb_start: float) -> float:
    """
    Return the LARGEST wb that still meets target_f3, searching over [0.34, wb_start].
    If current wb already meets target, return it.
    """
    cur = predict_f3_from_wb(wb_start, fam_key)
    if cur >= target_f3:
        return float(round(wb_start, 3))

    lo, hi = 0.34, max(0.34, min(0.75, wb_start))
    if predict_f3_from_wb(lo, fam_key) < target_f3:
        return float(round(lo, 3))

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if predict_f3_from_wb(mid, fam_key) >= target_f3:
            lo = mid
        else:
            hi = mid
    return float(round(hi, 3))

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
    """
    Blend data-driven inverse with curve; lean on data at low strengths.
    At <=40 MPa: ~70% data, >=60 MPa: ~40% data. Caps w/b at 0.75.
    """
    wb_curve = _wb_from_curve(f28, fam_key)
    wb_knn = _wb_from_knn(f28, fam_key)
    if wb_knn is None:
        wb = wb_curve
    else:
        f = float(f28)
        w_data = 0.7 if f <= 40 else (0.4 if f >= 60 else 0.7 - (0.3 * (f - 40) / 20.0))
        wb = w_data * wb_knn + (1.0 - w_data) * wb_curve
    return max(0.34, min(0.75, wb))

def implied_f28_from_wb(wb, fam_key):
    A, b = family_curves.get(fam_key, (A_g, b_g))
    return float(A * (wb ** b))

# ============================================================
# Water prediction model (used only when not fixed-water)
# ============================================================
family_models_water = {}
for fam in BINDER_FAMILIES.keys():
    sub = df[df["_family"] == fam]
    if len(sub) >= 2:
        family_models_water[fam] = KNeighborsRegressor(n_neighbors=3, weights="distance") \
            .fit(sub[["Strength3d", "Strength28d", "Water/Binder"]].values, sub["Free Water"].values)

knn_water_global = KNeighborsRegressor(n_neighbors=3, weights="distance") \
    .fit(df[["Strength3d", "Strength28d", "Water/Binder"]].values, df["Free Water"].values)

W_MIN, W_MAX = float(df["Free Water"].min()), float(df["Free Water"].max())

def get_water(f3, f28, wb, fam_key, use_fixed_water: bool, water_fixed: float):
    if use_fixed_water:
        return float(water_fixed)
    mdl_w = family_models_water.get(fam_key, knn_water_global)
    w = float(mdl_w.predict([[float(f3), float(f28), float(wb)]])[0])
    return float(max(W_MIN, min(W_MAX, w)))

# ============================================================
# Report-aligned EC method (A1 + A2 + A3)
# ============================================================
# Table 2.0.2 (A1): kgCO2 per kg material
A1_EF_PER_KG = {
    "Water": 0.00045,
    "Cement": 0.91796,
    "GGBFS": 0.19230,
    "Fly Ash": 0.0,
    "20mm Aggregate": 0.01045,
    "10mm Aggregate": 0.01045,
    "Man Sand": 0.00507,
    "Natural Sand": 0.00419,
    "Plastiment 30": 0.00167,
    "ECO WR": 0.00167,
    "Retarder": 0.00167,
}

# Table 2.0.3 (A2): transport EF (kgCO2/km) and assumed distances (km)
A2_TRUCK_EF_PER_KM = 0.000102576
A2_SEA_EF_PER_KM   = 0.00001321

A2_DIST = {
    "Water": {"truck_km": 0, "sea_km": 0},
    "Cement": {"truck_km": 310, "sea_km": 700},
    "GGBFS": {"truck_km": 310, "sea_km": 700},
    "Fly Ash": {"truck_km": 310, "sea_km": 700},
    "20mm Aggregate": {"truck_km": 40, "sea_km": 0},
    "10mm Aggregate": {"truck_km": 40, "sea_km": 0},
    "Man Sand": {"truck_km": 40, "sea_km": 0},
    "Natural Sand": {"truck_km": 40, "sea_km": 0},
    "Plastiment 30": {"truck_km": 1000, "sea_km": 8600},
    "ECO WR": {"truck_km": 1000, "sea_km": 8600},
    "Retarder": {"truck_km": 1000, "sea_km": 8600},
}

EC_A3_CONST = 6.9  # kgCO2 per m3

def compute_ec_report_aligned(water, binder_split, aggs, admix_tuple):
    quantities = {
        "Water": float(water),
        "Cement": float(binder_split.get("Cement", 0.0)),
        "GGBFS": float(binder_split.get("GGBFS", 0.0)),
        "Fly Ash": float(binder_split.get("Fly Ash", 0.0)),
        "20mm Aggregate": float(aggs.get("20mm Aggregate", 0.0)),
        "10mm Aggregate": float(aggs.get("10mm Aggregate", 0.0)),
        "Man Sand": float(aggs.get("Man Sand", 0.0)),
        "Natural Sand": float(aggs.get("Natural Sand", 0.0)),
    }
    plast, eco, ret = admix_tuple
    quantities["Plastiment 30"] = float(plast)
    quantities["ECO WR"] = float(eco)
    quantities["Retarder"] = float(ret)

    # A1
    ec_a1 = 0.0
    for mat, qty in quantities.items():
        ec_a1 += qty * A1_EF_PER_KG.get(mat, 0.0)

    # A2 (truck + sea)
    ec_a2 = 0.0
    for mat, qty in quantities.items():
        cfg = A2_DIST.get(mat)
        if not cfg:
            continue
        per_kg_transport = (
            A2_TRUCK_EF_PER_KM * cfg["truck_km"] +
            A2_SEA_EF_PER_KM * cfg["sea_km"]
        )
        ec_a2 += qty * per_kg_transport

    # A3
    ec_a3 = float(EC_A3_CONST)

    total = ec_a1 + ec_a2 + ec_a3
    return {"EC_A1": ec_a1, "EC_A2": ec_a2, "EC_A3": ec_a3, "EC_total": total}

# ============================================================
# Spreadsheet-identical paste/agg math helpers
# ============================================================
def _wb_band_split(wb: float):
    if wb >= 0.58:
        key = 0.58
    elif wb >= 0.50:
        key = 0.50
    elif wb >= 0.42:
        key = 0.42
    else:
        key = 0.34
    return SHEET_SPLITS[key]

def _paste_volume_liters(water_kg, c_kg, s_kg, fa_kg):
    return (water_kg / DENSITY["Water"]
            + c_kg / DENSITY["Cement"]
            + s_kg / DENSITY["GGBFS"]
            + fa_kg / DENSITY["Fly Ash"]
            + SHEET_PASTE_EXTRA_LITERS)

def _combined_agg_density_kg_per_L(split):
    return (split["20mm"] * DENSITY["20mm Aggregate"]
            + split["10mm"] * DENSITY["10mm Aggregate"]
            + split["Man"]  * DENSITY["Man Sand"]
            + split["Nat"]  * DENSITY["Natural Sand"])

def _family_uses_flyash(fam_key: str) -> bool:
    fam = BINDER_FAMILIES.get(fam_key.upper(), {})
    return (fam.get("Fly Ash", 0.0) or 0.0) > 0.0

def compute_admixtures_from_sheet(binder_total_kg_m3: float, fam_key: str):
    scale = binder_total_kg_m3 / 100.0
    ret_kg   = ADM_DOSE_MASS_PER_100KG["Retarder"] * scale
    plast_kg = ADM_DOSE_MASS_PER_100KG["Plastiment 30"] * scale
    eco_key  = "ECO_WR_FA_OR_TERNARY" if _family_uses_flyash(fam_key) else "ECO_WR_PC_OR_SLAG"
    eco_kg   = ADM_DOSE_MASS_PER_100KG[eco_key] * scale
    return plast_kg, eco_kg, ret_kg

# ============================================================
# Core design function
# ============================================================
def design_mix_from_strengths_min(
    f3_min,
    f28_min,
    binder_family_key="S5",
    use_fixed_water=True,
    water_fixed=WATER_DEFAULT,
    use_manual_wb=False,
    manual_wb_value=0.50,
    enforce_3d=False
):
    fam_key = binder_family_key.upper()

    # 1) w/b from f28 (blended inverse) OR manual override
    if use_manual_wb:
        wb_pred = float(manual_wb_value)
    else:
        wb_pred = float(predict_wb_from_f28_curve(f28_min, fam_key))

    # Optional: enforce 3-day minimum by tightening w/b (only when not manual)
    wb_before_3d = wb_pred
    if (not use_manual_wb) and enforce_3d:
        wb_pred = float(recommend_wb_for_f3(float(f3_min), fam_key, wb_pred))

    # 2) Water (fixed or predicted)
    water_pred = float(get_water(f3_min, f28_min, wb_pred, fam_key, use_fixed_water, water_fixed))

    # 3) Binder total + split per family
    binder_total = float(water_pred / wb_pred)
    fam = BINDER_FAMILIES[fam_key]
    slag_frac, fly_frac = float(fam.get("GGBFS", 0.0)), float(fam.get("Fly Ash", 0.0))
    cem_frac = max(0.0, 1.0 - slag_frac - fly_frac)
    c, s, fa = binder_total * cem_frac, binder_total * slag_frac, binder_total * fly_frac

    # 4) Admixtures (sheet-accurate)
    plast, eco, ret = compute_admixtures_from_sheet(binder_total, fam_key)
    adm_total = plast + eco + ret

    # 5) Aggregates from sheet logic
    split = _wb_band_split(wb_pred)
    V_paste_L = _paste_volume_liters(water_pred, c, s, fa)
    air_pct = float(DEFAULT_AIR_PERCENT)
    V_air_L = (air_pct / 100.0) * 1000.0
    V_agg_L = max(0.0, 1000.0 - V_paste_L - (V_air_L if SHEET_SUBTRACT_AIR else 0.0))
    rho_agg_kg_per_L = _combined_agg_density_kg_per_L(split)

    M_agg_total = V_agg_L * rho_agg_kg_per_L
    a20 = M_agg_total * split["20mm"]
    a10 = M_agg_total * split["10mm"]
    ms  = M_agg_total * split["Man"]
    ns  = M_agg_total * split["Nat"]

    rho_target = water_pred + binder_total + adm_total + M_agg_total

    # 6) EC (REPORT-ALIGNED)
    ec_breakdown = compute_ec_report_aligned(
        water_pred,
        {"Cement": c, "GGBFS": s, "Fly Ash": fa},
        {"20mm Aggregate": a20, "10mm Aggregate": a10, "Man Sand": ms, "Natural Sand": ns},
        (plast, eco, ret)
    )

    total_mass = water_pred + binder_total + (a20 + a10 + ms + ns) + adm_total

    return {
        "inputs": {
            "min_3d_MPa": float(f3_min),
            "min_28d_MPa": float(f28_min),
            "binder_family": fam_key,
            "use_fixed_water": bool(use_fixed_water),
            "water_fixed": float(water_fixed),
            "use_manual_wb": bool(use_manual_wb),
            "manual_wb_value": float(manual_wb_value) if use_manual_wb else None,
            "enforce_3d": bool(enforce_3d),
        },
        "predicted_parameters": {
            "water_binder_ratio": float(wb_pred),
            "water_binder_ratio_before_3d": float(wb_before_3d),
            "water_kg_m3": float(water_pred),
            "binder_total_kg_m3": float(binder_total),
            "fresh_density_target_kg_m3": float(rho_target),
            "air_percent": float(air_pct),
        },
        "binder_exact": {"Cement": float(c), "GGBFS": float(s), "Fly Ash": float(fa)},
        "admixture_split_kg_m3": {"Plastiment 30": float(plast), "ECO WR": float(eco), "Retarder": float(ret)},
        "aggregates_exact": {"20mm Aggregate": float(a20), "10mm Aggregate": float(a10), "Man Sand": float(ms), "Natural Sand": float(ns)},
        "embodied_carbon": ec_breakdown,
        "totals": {"sum_all_components_kg_m3": float(total_mass)}
    }

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Inputs")

    use_fixed_water = st.toggle("Use fixed water (validation mode)", value=True)
    water_fixed = st.number_input("Water (kg/m³)", 120.0, 240.0, WATER_DEFAULT, 1.0, disabled=not use_fixed_water)

    fam_keys = list(BINDER_FAMILIES.keys())
    default_idx = fam_keys.index("S5") if "S5" in fam_keys else 0
    fam = st.selectbox("Binder family", fam_keys, index=default_idx)

    st.divider()

    use_manual_wb = st.checkbox("Set w/b manually (ignores strength targets)")
    enforce_3d = st.checkbox("Enforce 3-day minimum (tighten w/b if needed)", value=False, disabled=use_manual_wb)

    colA, colB = st.columns(2)
    with colA:
        f3_min = st.number_input("3-day strength target (MPa)", 5.0, 80.0, 30.0, 0.5, disabled=use_manual_wb)
    with colB:
        f28_min = st.number_input("28-day strength target (MPa)", 10.0, 100.0, 50.0, 0.5, disabled=use_manual_wb)

    wb_manual = st.number_input("Manual w/b ratio", 0.30, 0.80, 0.45, 0.01, disabled=not use_manual_wb)

    run_btn = st.button("Design mix", type="primary", use_container_width=True)

# ============================================================
# Binder family mixes panel
# ============================================================
def _family_percent_rows():
    rows = []
    for k, v in BINDER_FAMILIES.items():
        slag = 100.0 * v.get("GGBFS", 0.0)
        fly  = 100.0 * v.get("Fly Ash", 0.0)
        cem  = max(0.0, 100.0 - slag - fly)
        rows.append((k, cem, slag, fly))
    order = ["P1","F2","F4","S3","S5","S6","T1","T2"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 999)
    return rows

with st.expander("Binder family mixes (percent of total binder)", expanded=True):
    fam_table = pd.DataFrame(_family_percent_rows(), columns=["Family", "Cement %", "GGBFS %", "Fly Ash %"])
    st.dataframe(fam_table, use_container_width=True, hide_index=True)

# ============================================================
# Main UI helpers
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

    df_rows, total_mass, total_vol = [], 0.0, 0.0
    for name, dens, mass in rows:
        vol = mass / dens if dens > 0 else 0.0
        df_rows.append([name, dens, mass, vol])
        total_mass += mass
        total_vol += vol

    # Air row
    air_vol_L = (out["predicted_parameters"]["air_percent"] / 100.0) * 1000.0
    df_rows.append(["Air", np.nan, 0.0, air_vol_L])
    total_vol += air_vol_L

    df_out = pd.DataFrame(df_rows, columns=["Material", "Density (kg/L)", "Mass (kg)", "Volume (L)"])
    df_out["Mass %"] = 100.0 * df_out["Mass (kg)"] / total_mass

    # format
    df_out["Density (kg/L)"] = df_out["Density (kg/L)"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    df_out["Mass (kg)"] = df_out["Mass (kg)"].map(lambda x: f"{x:.2f}")
    df_out["Volume (L)"] = df_out["Volume (L)"].map(lambda x: f"{x:.2f}")
    df_out["Mass %"] = df_out["Mass %"].map(lambda x: f"{x:.2f}")

    avg_density = total_mass / total_vol if total_vol else 0.0
    df_total = pd.DataFrame(
        [["Total (avg density)", f"{avg_density:.2f}", f"{total_mass:.2f}", f"{total_vol:.2f}", ""]],
        columns=df_out.columns
    )
    return pd.concat([df_out, df_total], ignore_index=True)

# ============================================================
# Run
# ============================================================
if run_btn:
    out = design_mix_from_strengths_min(
        f3_min=f3_min,
        f28_min=f28_min,
        binder_family_key=fam,
        use_fixed_water=use_fixed_water,
        water_fixed=water_fixed if use_fixed_water else WATER_DEFAULT,
        use_manual_wb=use_manual_wb,
        manual_wb_value=wb_manual,
        enforce_3d=enforce_3d,
    )

    pp = out["predicted_parameters"]
    wb = pp["water_binder_ratio"]
    wb_before_3d = pp["water_binder_ratio_before_3d"]
    water = pp["water_kg_m3"]
    btot = pp["binder_total_kg_m3"]

    ec = out["embodied_carbon"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Binder family", out["inputs"]["binder_family"])

    if out["inputs"]["use_manual_wb"]:
        m2.metric("w/b (manual)", f"{wb:.3f}")
    else:
        if out["inputs"]["enforce_3d"] and abs(wb - wb_before_3d) > 1e-6:
            m2.metric("w/b (28d + 3d enforced)", f"{wb:.3f}", delta=f"{(wb - wb_before_3d):.3f}")
        else:
            m2.metric("w/b (curve+data)", f"{wb:.3f}")

    m3.metric("Water (kg/m³)", f"{water:.1f}" + (" (fixed)" if out["inputs"]["use_fixed_water"] else " (predicted)"))
    m4.metric("Binder total (kg/m³)", f"{btot:.1f}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sum Concrete (kg/m³)", f"{pp['fresh_density_target_kg_m3']:.1f}")
    c2.metric("Air (%)", f"{pp['air_percent']:.1f}")
    c3.metric("EC total (kg CO₂e/m³)", f"{ec['EC_total']:.1f}")
    c4.metric("A1 / A2 / A3", f"{ec['EC_A1']:.1f} / {ec['EC_A2']:.1f} / {ec['EC_A3']:.1f}")

    # 3-day reporting
    f3_pred_report = predict_f3_from_wb(wb, fam)
    if out["inputs"]["use_manual_wb"]:
        st.info(f"Predicted 3-day ≈ **{f3_pred_report:.1f} MPa** at w/b = {wb:.3f} (targets ignored)")
        f28_pred_report = implied_f28_from_wb(wb, fam)
        st.success(f"Predicted 28-day ≈ **{f28_pred_report:.1f} MPa** at w/b = {wb:.3f} (targets ignored)")
    else:
        msg = f"Predicted 3-day ≈ **{f3_pred_report:.1f} MPa** at w/b = {wb:.3f} (target: {float(f3_min):.1f} MPa)"
        if f3_pred_report + 1e-9 < float(f3_min) and not enforce_3d:
            wb_rec = recommend_wb_for_f3(float(f3_min), fam, wb)
            f3_at = predict_f3_from_wb(wb_rec, fam)
            f28_at = implied_f28_from_wb(wb_rec, fam)
            msg += f"\n\nSuggestion: tighten to **w/b ≈ {wb_rec:.3f}** (predicts 3d ≈ {f3_at:.1f} MPa, 28d ≈ {f28_at:.1f} MPa)."
        st.info(msg)

    st.subheader("Detailed Materials Table")
    st.dataframe(material_table(out), use_container_width=True, hide_index=True)

    st.subheader("Aggregates (kg/m³)")
    aggs_df = pd.DataFrame({
        "Aggregate": list(out["aggregates_exact"].keys()),
        "kg/m³": [f"{v:.2f}" for v in out["aggregates_exact"].values()]
    })
    st.dataframe(aggs_df, use_container_width=True, hide_index=True)

    st.subheader("Binder Split")
    b = out["binder_exact"]
    binder_split_df = pd.DataFrame({
        "Component": ["Cement", "GGBFS", "Fly Ash"],
        "kg/m³": [b["Cement"], b["GGBFS"], b["Fly Ash"]]
    })
    binder_split_df["% of binder"] = 100 * binder_split_df["kg/m³"] / btot
    binder_split_df["kg/m³"] = binder_split_df["kg/m³"].map(lambda x: f"{x:.2f}")
    binder_split_df["% of binder"] = binder_split_df["% of binder"].map(lambda x: f"{x:.2f}")
    st.dataframe(binder_split_df, use_container_width=True, hide_index=True)

    with st.expander("Embodied carbon breakdown (report method)", expanded=False):
        st.write({
            "EC_A1": float(ec["EC_A1"]),
            "EC_A2": float(ec["EC_A2"]),
            "EC_A3": float(ec["EC_A3"]),
            "EC_total": float(ec["EC_total"]),
        })

    st.divider()
