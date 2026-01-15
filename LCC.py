# LCC.py
from __future__ import annotations

from lcc.config import USE_FIXED_WATER
from lcc.dataset import build_df
from lcc.models import build_models, predict_f3_from_wb, predict_f7_from_wb, implied_f28_from_wb
from lcc.design import design_mix_from_strengths_min
from lcc.cli import (
    choose_family_with_percentages,
    choose_early_age_and_min_strength,
    ask_float,
    print_binder_split_display,
    render_detailed_table,
)

def main():
    df = build_df()
    models = build_models(df)

    mode = "VALIDATION (Water fixed at 190)" if USE_FIXED_WATER else "DESIGN (Water predicted)"
    print(f"\n=== Strength-Driven Mix Designer — {mode}; Curve-based w/b; early-age input can be 3d or 7d; REPORT-aligned EC (A1+A2+A3) ===")

    fam = choose_family_with_percentages(default_key="S5")
    early_age_days, early_min = choose_early_age_and_min_strength()
    f28_min = ask_float("Minimum 28-day strength (MPa)", 50)

    out = design_mix_from_strengths_min(models, early_min, f28_min, fam, early_age_days=early_age_days)

    pp = out["predicted_parameters"]
    wb = pp["water_binder_ratio"]
    water = pp["water_kg_m3"]
    btot = pp["binder_total_kg_m3"]

    print("\n=== Predicted Mix Design ===")
    print(f"Family: {out['inputs']['binder_family']}")
    print(f"Early-age requirement: ≥ {out['inputs']['min_early_MPa']:.1f} MPa at {out['inputs']['early_age_days']} days")
    print(f"28-day requirement:    ≥ {out['inputs']['min_28d_MPa']:.1f} MPa")

    print(f"\nw/b (curve+data) ≈ {wb:.3f} | Water = {water:.1f} kg/m³ | Binder ≈ {btot:.1f} kg/m³")

    # Reporting only
    f3_pred_report = predict_f3_from_wb(models, wb, fam)
    f7_pred_report = predict_f7_from_wb(models, wb, fam)
    f28_pred_report = implied_f28_from_wb(models, wb, fam)
    print("\nPredicted strengths (reporting only):")
    print(f"  f3  ≈ {f3_pred_report:.1f} MPa")
    print(f"  f7  ≈ {f7_pred_report:.1f} MPa")
    print(f"  f28 ≈ {f28_pred_report:.1f} MPa")

    print_binder_split_display(out["binder_exact"], btot, step=1.0)

    print("\nAggregates (kg/m³):")
    for k, v in out["aggregates_exact"].items():
        print(f"  {k:<15} {v:>8.1f}")

    print(f"\nSum Concrete (kg/m³): {pp['fresh_density_target_kg_m3']:.1f}")

    ec = out["embodied_carbon"]
    print(f"\nEmbodied carbon (kg CO2e/m³) — report method:")
    print(f"  EC_A1:    {ec['EC_A1']:.2f}")
    print(f"  EC_A2:    {ec['EC_A2']:.2f}")
    print(f"  EC_A3:    {ec['EC_A3']:.2f}")
    print(f"  EC_total: {ec['EC_total']:.2f}")

    render_detailed_table(out)

if __name__ == "__main__":
    main()
