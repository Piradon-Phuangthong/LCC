# lcc/ec.py
from __future__ import annotations

from .config import A1_EF_PER_KG, A2_DIST, A2_TRUCK_EF_PER_KM, A2_SEA_EF_PER_KM, EC_A3_CONST

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

    ec_a1 = 0.0
    for mat, qty in quantities.items():
        ec_a1 += qty * A1_EF_PER_KG.get(mat, 0.0)

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

    ec_a3 = float(EC_A3_CONST)
    total = ec_a1 + ec_a2 + ec_a3
    return {"EC_A1": ec_a1, "EC_A2": ec_a2, "EC_A3": ec_a3, "EC_total": total}
