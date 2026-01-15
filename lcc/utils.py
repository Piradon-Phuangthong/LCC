# lcc/utils.py
from __future__ import annotations

from .config import ADM_DOSE_MASS_PER_100KG, BINDER_FAMILIES

def _family_uses_flyash(fam_key: str) -> bool:
    fam = BINDER_FAMILIES.get(fam_key.upper(), {})
    return (fam.get("Fly Ash", 0.0) or 0.0) > 0.0

def compute_admixtures_from_sheet(binder_total_kg_m3: float, fam_key: str):
    scale = binder_total_kg_m3 / 100.0

    ret_kg   = ADM_DOSE_MASS_PER_100KG["Retarder"]      * scale
    plast_kg = ADM_DOSE_MASS_PER_100KG["Plastiment 30"] * scale

    eco_key = "ECO_WR_FA_OR_TERNARY" if _family_uses_flyash(fam_key) else "ECO_WR_PC_OR_SLAG"
    eco_kg  = ADM_DOSE_MASS_PER_100KG[eco_key] * scale

    return plast_kg, eco_kg, ret_kg
