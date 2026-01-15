# lcc/aggregates.py
from __future__ import annotations

from .config import (
    DENSITY,
    SHEET_PASTE_EXTRA_LITERS,
    SHEET_SPLITS,
    SHEET_SPLIT_HIGH_WB,
    WB_FAMILIES_TEMPLATES,
)

def wb_band_split(wb: float):
    """Choose band exactly like the sheet: ≥0.58, 0.50–<0.58, 0.42–<0.50, <0.42."""
    if wb >= 0.58:
        key = 0.58
    elif wb >= 0.50:
        key = 0.50
    elif wb >= 0.42:
        key = 0.42
    else:
        key = 0.34
    return SHEET_SPLITS[key]

def paste_volume_liters(water_kg: float, c_kg: float, s_kg: float, fa_kg: float, densities=DENSITY) -> float:
    return (
        water_kg/densities["Water"]
        + c_kg/densities["Cement"]
        + s_kg/densities["GGBFS"]
        + fa_kg/densities["Fly Ash"]
        + SHEET_PASTE_EXTRA_LITERS
    )

def combined_agg_density_kg_per_L(split, densities=DENSITY) -> float:
    return (
        split["20mm"]*densities["20mm Aggregate"]
        + split["10mm"]*densities["10mm Aggregate"]
        + split["Man"] *densities["Man Sand"]
        + split["Nat"] *densities["Natural Sand"]
    )

def nearest_template_family(wb: float):
    return min(WB_FAMILIES_TEMPLATES, key=lambda t: abs(t[0] - wb))

def normalized_split_from_template(center_wb: float):
    for cw, (a20, a10, ms, ns), _, _ in WB_FAMILIES_TEMPLATES:
        if abs(cw - center_wb) < 1e-6:
            s = a20 + a10 + ms + ns
            return {"20mm": a20/s, "10mm": a10/s, "Man": ms/s, "Nat": ns/s}
    return SHEET_SPLIT_HIGH_WB
