# lcc/config.py
from __future__ import annotations

# ============================================================
# -------- Configuration: toggle validation vs design --------
# ============================================================
USE_FIXED_WATER = True    # True = validation mode (fixed water); False = design mode (predict water)
WATER_FIXED = 190.0       # kg/m³ when validation mode is ON

# Reproduce spreadsheet aggregate math (V_paste -> V_agg -> M_agg -> split)
SHEET_METHOD = True
SHEET_PASTE_EXTRA_LITERS = 15.0  # the +15 L in your sheet

# Excel uses N = 1000 - M (does NOT subtract air before sizing aggregates)
SHEET_SUBTRACT_AIR = False  # set True only if you want to subtract air pre-aggregates

# ============================================================
# Material densities (kg/L)
# ============================================================
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

# From your yellow table (rates)
SHEET_SPLITS = {
    0.58: {"20mm": 0.392, "10mm": 0.168, "Nat": 0.264, "Man": 0.176},
    0.50: {"20mm": 0.406, "10mm": 0.174, "Nat": 0.252, "Man": 0.168},
    0.42: {"20mm": 0.420, "10mm": 0.180, "Nat": 0.240, "Man": 0.160},
    0.34: {"20mm": 0.448, "10mm": 0.192, "Nat": 0.216, "Man": 0.144},
}
SHEET_SPLIT_HIGH_WB = SHEET_SPLITS[0.58]

# ============================================================
# Binder families (mass fractions of binder)
# ============================================================
BINDER_FAMILIES = {
    "P1": {"GGBFS": 0.00, "Fly Ash": 0.00},  # 100% PC
    "F2": {"GGBFS": 0.00, "Fly Ash": 0.25},  # 25% Fly
    "F4": {"GGBFS": 0.00, "Fly Ash": 0.40},  # 40% Fly
    "F5": {"GGBFS": 0.00, "Fly Ash": 0.50},  # 50% Fly
    "S3": {"GGBFS": 0.35, "Fly Ash": 0.00},  # 35% Slag
    "S5": {"GGBFS": 0.50, "Fly Ash": 0.00},  # 50% Slag
    "S6": {"GGBFS": 0.65, "Fly Ash": 0.00},  # 65% Slag
    "T1": {"GGBFS": 0.40, "Fly Ash": 0.20},  # 40% Slag + 20% Fly
    "T2": {"GGBFS": 0.40, "Fly Ash": 0.30},  # 40% Slag + 30% Fly
    "T3": {"GGBFS": 0.30, "Fly Ash": 0.30},  # 30% Slag + 30% Fly
}
FAMILY_PROTOTYPES = {k: (v["GGBFS"], v["Fly Ash"]) for k, v in BINDER_FAMILIES.items()}

# -------- Admixture doses from your sheet (per 100 kg binder) --------
ADM_DOSE_MASS_PER_100KG = {
    "Retarder":      0.105,  # All binders
    "Plastiment 30": 0.315,  # All binders (low-range WR)
    "ECO_WR_FA_OR_TERNARY": 0.214,  # FA & ternary (F2, F4, F5, T1, T2, T3)
    "ECO_WR_PC_OR_SLAG":    0.535,  # GP (P1) & Slag blends (S3, S5, S6)
}

# ============================================================
# Embodied carbon method — REPORT-ALIGNED (A1 + A2 + A3)
# ============================================================
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

EC_A3_CONST = 6.9  # kgCO2e per m3

# ============================================================
# Aggregate templates by w/b family (unchanged fallback)
# ============================================================
WB_FAMILIES_TEMPLATES = [
    (0.57, (726, 318, 333, 497), 2407, 1.8),
    (0.50, (720, 315, 315, 462), 2390, 1.8),
    (0.42, (718, 314, 294, 441), 2420, 1.9),
    (0.34, (727, 318, 252, 374), 2442, 1.4),
    (0.66, (721, 317, 346, 516), 2392, 1.9),
]
