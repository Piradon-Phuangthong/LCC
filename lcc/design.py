# lcc/design.py
from __future__ import annotations

from typing import Dict, Any, Optional

from .config import (
    BINDER_FAMILIES,
    SHEET_METHOD,
    SHEET_SUBTRACT_AIR,
)
from .models import ModelsBundle, predict_wb_from_f28_curve, get_water
from .aggregates import (
    wb_band_split,
    paste_volume_liters,
    combined_agg_density_kg_per_L,
    nearest_template_family,
)
from .ec import compute_ec_report_aligned
from .utils import compute_admixtures_from_sheet


def design_mix_from_strengths_min(
    models: ModelsBundle,
    early_min: float,
    f28_min: float,
    binder_family_key: str = "S5",
    early_age_days: int = 3,
    wb_override: Optional[float] = None,  # <<< NEW
) -> Dict[str, Any]:
    fam_key = binder_family_key.upper()

    # 1) w/b from f28 (blended inverse) OR override
    if wb_override is not None:
        wb_pred = float(wb_override)
    else:
        wb_pred = predict_wb_from_f28_curve(models, f28_min, fam_key)

    # 2) Water (fixed or predicted)
    # Note: if you're in VALIDATION mode, water is fixed anyway.
    # If you're in DESIGN mode, water model uses (early_strength, f28, wb).
    water_pred = get_water(
        models,
        early_min,
        f28_min,
        wb_pred,
        fam_key,
        early_age_days=int(early_age_days),
    )

    # 3) Binder total + split per chosen family
    binder_total = water_pred / wb_pred
    fam = BINDER_FAMILIES[fam_key]
    slag_frac, fly_frac = fam.get("GGBFS", 0.0), fam.get("Fly Ash", 0.0)
    cem_frac = max(0.0, 1.0 - slag_frac - fly_frac)
    c, s, fa = binder_total * cem_frac, binder_total * slag_frac, binder_total * fly_frac

    # 4) Admixtures — per 100 kg binder (sheet-accurate)
    plast, eco, ret = compute_admixtures_from_sheet(binder_total, fam_key)
    adm_total = plast + eco + ret

    # 5) Aggregates from sheet logic (or fallback to template scaling)
    if SHEET_METHOD:
        split = wb_band_split(wb_pred)
        V_paste_L = paste_volume_liters(water_pred, c, s, fa)

        air_pct = 1.9
        V_air_L = (air_pct / 100.0) * 1000.0
        V_agg_L = max(
            0.0,
            1000.0 - V_paste_L - (V_air_L if SHEET_SUBTRACT_AIR else 0.0),
        )

        rho_agg_kg_per_L = combined_agg_density_kg_per_L(split)

        M_agg_total = V_agg_L * rho_agg_kg_per_L
        a20 = M_agg_total * split["20mm"]
        a10 = M_agg_total * split["10mm"]
        ms = M_agg_total * split["Man"]
        ns = M_agg_total * split["Nat"]

        rho_target = water_pred + binder_total + adm_total + M_agg_total
        center_wb = wb_pred
    else:
        center_wb, (a20_b, a10_b, ms_b, ns_b), rho_target, air_pct = nearest_template_family(wb_pred)
        base_ag = a20_b + a10_b + ms_b + ns_b
        non_ag = water_pred + binder_total + adm_total
        aggs_needed = max(0.0, rho_target - non_ag)
        scale = aggs_needed / base_ag if base_ag > 0 else 1.0
        a20, a10, ms, ns = a20_b * scale, a10_b * scale, ms_b * scale, ns_b * scale

    aggs_dict = {
        "20mm Aggregate": a20,
        "10mm Aggregate": a10,
        "Man Sand": ms,
        "Natural Sand": ns,
    }
    binder_dict = {"Cement": c, "GGBFS": s, "Fly Ash": fa}

    # 6) EC (REPORT-ALIGNED)
    ec_breakdown = compute_ec_report_aligned(water_pred, binder_dict, aggs_dict, (plast, eco, ret))
    total_mass = water_pred + binder_total + (a20 + a10 + ms + ns) + adm_total

    return {
        "inputs": {
            "min_early_MPa": float(early_min),
            "early_age_days": int(early_age_days),
            "min_28d_MPa": float(f28_min),
            "binder_family": fam_key,
            "wb_override": None if wb_override is None else float(wb_override),  # <<< optional tracking
        },
        "predicted_parameters": {
            "water_binder_ratio": float(wb_pred),
            "water_kg_m3": float(water_pred),
            "binder_total_kg_m3": float(binder_total),
            "fresh_density_target_kg_m3": float(rho_target),
            "air_percent": float(air_pct),
            "wb_family_center": float(center_wb),
        },
        "binder_exact": binder_dict,
        "admixture_split_kg_m3": {"Plastiment 30": plast, "ECO WR": eco, "Retarder": ret},
        "aggregates_exact": aggs_dict,
        "embodied_carbon": ec_breakdown,
        "totals": {"sum_all_components_kg_m3": float(total_mass)},
    }
