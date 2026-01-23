# LCC.py
from __future__ import annotations

from lcc.config import USE_FIXED_WATER
from lcc.dataset import build_df
from lcc.models import (
    build_models,
    predict_f3_from_wb,
    predict_f7_from_wb,
    implied_f28_from_wb,
    predict_wb_from_f28_curve,
)
from lcc.design import design_mix_from_strengths_min
from lcc.cli import (
    choose_family_with_percentages,
    choose_early_age_and_min_strength,
    ask_float,
    print_binder_split_display,
    render_detailed_table,
)


def _ask_yes_no(prompt: str, default: bool = False) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        s = input(f"{prompt} ({d}): ").strip().lower()
        if not s:
            return default
        if s in ("y", "yes"):
            return True
        if s in ("n", "no"):
            return False
        print("Please enter y or n.")


def _ask_wb_override(suggested_wb: float) -> float | None:
    use_override = _ask_yes_no("Override w/b?", default=False)
    if not use_override:
        return None

    while True:
        s = input(f"Enter w/b override [suggested {suggested_wb:.3f}]: ").strip()
        if not s:
            return float(suggested_wb)
        try:
            wb = float(s)
        except ValueError:
            print("Please enter a number (e.g. 0.72).")
            continue

        if wb <= 0.0:
            print("w/b must be > 0.")
            continue
        if not (0.20 <= wb <= 1.20):
            print("That w/b looks unrealistic. Try something like 0.30–0.90.")
            continue

        return float(wb)


def main():
    df = build_df()
    models = build_models(df)

    mode = "VALIDATION (Water fixed at 190)" if USE_FIXED_WATER else "DESIGN (Water predicted)"

    while True:
        print(
            f"\n=== Strength-Driven Mix Designer — {mode}; "
            f"Curve-based w/b; early-age input can be 3d or 7d; "
            f"REPORT-aligned EC (A1+A2+A3) ==="
        )

        fam = choose_family_with_percentages(default_key="S5")
        early_age_days, early_min = choose_early_age_and_min_strength()
        f28_min = ask_float("Minimum 28-day strength (MPa)", 50)

        # Suggested w/b from model
        suggested_wb = predict_wb_from_f28_curve(models, f28_min, fam)
        wb_override = _ask_wb_override(suggested_wb)

        out = design_mix_from_strengths_min(
            models,
            early_min,
            f28_min,
            fam,
            early_age_days=early_age_days,
            wb_override=wb_override,
        )

        pp = out["predicted_parameters"]
        wb = pp["water_binder_ratio"]
        water = pp["water_kg_m3"]
        btot = pp["binder_total_kg_m3"]

        print("\n=== Predicted Mix Design ===")
        print(f"Family: {out['inputs']['binder_family']}")
        print(
            f"Early-age requirement: ≥ {out['inputs']['min_early_MPa']:.1f} MPa "
            f"at {out['inputs']['early_age_days']} days"
        )
        print(f"28-day requirement:    ≥ {out['inputs']['min_28d_MPa']:.1f} MPa")

        if wb_override is not None:
            print(
                f"\nw/b OVERRIDE = {wb_override:.3f} | "
                f"Water = {water:.1f} kg/m³ | Binder ≈ {btot:.1f} kg/m³"
            )
        else:
            print(
                f"\nw/b (curve+data) ≈ {wb:.3f} | "
                f"Water = {water:.1f} kg/m³ | Binder ≈ {btot:.1f} kg/m³"
            )

        # Reporting only
        f3_pred = predict_f3_from_wb(models, wb, fam)
        f7_pred = predict_f7_from_wb(models, wb, fam)
        f28_pred = implied_f28_from_wb(models, wb, fam)

        print("\nPredicted strengths (reporting only):")
        print(f"  f3  ≈ {f3_pred:.1f} MPa")
        print(f"  f7  ≈ {f7_pred:.1f} MPa")
        print(f"  f28 ≈ {f28_pred:.1f} MPa")

        print_binder_split_display(out["binder_exact"], btot, step=1.0)

        print("\nAggregates (kg/m³):")
        for k, v in out["aggregates_exact"].items():
            print(f"  {k:<15} {v:>8.1f}")

        print(f"\nSum Concrete (kg/m³): {pp['fresh_density_target_kg_m3']:.1f}")

        ec = out["embodied_carbon"]
        print("\nEmbodied carbon (kg CO2e/m³) — report method:")
        print(f"  EC_A1:    {ec['EC_A1']:.2f}")
        print(f"  EC_A2:    {ec['EC_A2']:.2f}")
        print(f"  EC_A3:    {ec['EC_A3']:.2f}")
        print(f"  EC_total: {ec['EC_total']:.2f}")

        render_detailed_table(out)

        # 🔁 Repeat?
        if not _ask_yes_no("\nRun another mix?", default=True):
            print("\nExiting mix designer.")
            break


if __name__ == "__main__":
    main()
