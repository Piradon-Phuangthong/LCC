# lcc/cli.py
from __future__ import annotations

from typing import Tuple

from .config import BINDER_FAMILIES, DENSITY


def _family_percent_rows():
    rows = []
    for k, v in BINDER_FAMILIES.items():
        slag = 100.0 * v.get("GGBFS", 0.0)
        fly  = 100.0 * v.get("Fly Ash", 0.0)
        cem  = max(0.0, 100.0 - slag - fly)
        rows.append((k, cem, slag, fly))

    # ✅ T3 removed from menu order
    order = ["P1", "F2", "F4", "F5", "S3", "S5", "S6", "T1", "T2"]
    rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 999)
    return rows


def show_binder_family_options():
    print("\n=== Binder Family Options (percent of total binder) ===")
    print(f"{'#':<2} {'Family':<6} {'Cement %':>9} {'GGBFS %':>9} {'Fly Ash %':>10}  Example")

    # ✅ T3 removed from examples
    example = {
        "P1": "100% PC",
        "F2": "25% Fly Ash",
        "F4": "40% Fly Ash",
        "F5": "50% Fly Ash",
        "S3": "35% Slag",
        "S5": "50% Slag",
        "S6": "65% Slag",
        "T1": "40% Slag + 20% Fly",
        "T2": "40% Slag + 30% Fly",
    }

    rows = _family_percent_rows()
    for i, (k, c, s, f) in enumerate(rows, start=1):
        print(f"{i:<2} {k:<6} {c:>8.0f}% {s:>9.0f}% {f:>10.0f}%  {example.get(k, '')}")
    return [k for k, _, _, _ in rows]


def choose_family_with_percentages(default_key="S5"):
    options = show_binder_family_options()
    default_idx = options.index(default_key) + 1 if default_key in options else 1
    s = input(f"\nChoose binder family (1-{len(options)} or code) [{default_idx}]: ").strip().upper()
    if not s:
        return options[default_idx - 1]
    if s.isdigit():
        i = int(s)
        if 1 <= i <= len(options):
            return options[i - 1]
        raise ValueError("Invalid number.")
    if s in options:
        return s
    raise ValueError(f"Please enter 1–{len(options)} or one of: {', '.join(options)}")


def ask_float(prompt, default=None):
    s = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not s and default is not None:
        return float(default)
    return float(s)


def choose_early_age_and_min_strength() -> Tuple[int, float]:
    s = input("\nEarly-age strength age? (3 or 7) [3]: ").strip()
    age = 3 if not s else int(float(s))
    if age not in (3, 7):
        raise ValueError("Please enter 3 or 7.")
    default_strength = 30.0 if age == 3 else 40.0
    early_min = ask_float(f"Minimum {age}-day strength (MPa)", default_strength)
    return age, float(early_min)


def print_binder_split_display(binder_exact, binder_total, step=1.0):
    c = float(binder_exact.get("Cement", 0.0))
    s = float(binder_exact.get("GGBFS", 0.0))
    fa = float(binder_exact.get("Fly Ash", 0.0))

    def snap(p, stp):
        return round(p / stp) * stp

    rows = [
        ("Cement", c, 100 * c / binder_total if binder_total else 0.0),
        ("GGBFS", s, 100 * s / binder_total if binder_total else 0.0),
        ("Fly Ash", fa, 100 * fa / binder_total if binder_total else 0.0),
    ]
    shown = [(n, round(m, 1), snap(p, step)) for (n, m, p) in rows]

    residual = round(100.0 - sum(p for _, _, p in shown), 10)
    imax = max(range(len(rows)), key=lambda i: rows[i][2])
    n, m, p = shown[imax]
    shown[imax] = (n, m, round(p + residual, 10))

    print("\nBinder split (kg/m³ and % of binder):")
    for n, m, p in shown:
        print(f"  {n:<12} {m:>8.1f} kg/m³  ({p:>3.0f}%)")
    print(f"  {'Binder Total':<12} {binder_total:>8.1f} kg/m³  (100%)")


def render_detailed_table(out, densities=DENSITY):
    w = out["predicted_parameters"]["water_kg_m3"]
    btot = out["predicted_parameters"]["binder_total_kg_m3"]
    air_pct = out["predicted_parameters"]["air_percent"]

    b = out["binder_exact"]
    a = out["aggregates_exact"]
    adm = out["admixture_split_kg_m3"]

    rows = [
        ("Water", densities["Water"], w),
        ("Cement", densities["Cement"], b.get("Cement", 0.0)),
        ("Fly Ash", densities["Fly Ash"], b.get("Fly Ash", 0.0)),
        ("GGBFS", densities["GGBFS"], b.get("GGBFS", 0.0)),
        ("20mm Aggregate", densities["20mm Aggregate"], a.get("20mm Aggregate", 0.0)),
        ("10mm Aggregate", densities["10mm Aggregate"], a.get("10mm Aggregate", 0.0)),
        ("Man Sand", densities["Man Sand"], a.get("Man Sand", 0.0)),
        ("Natural Sand", densities["Natural Sand"], a.get("Natural Sand", 0.0)),
        ("Plastiment 30", densities["Plastiment 30"], adm.get("Plastiment 30", 0.0)),
        ("ECO WR", densities["ECO WR"], adm.get("ECO WR", 0.0)),
        ("Retarder", densities["Retarder"], adm.get("Retarder", 0.0)),
    ]

    def vol_l(m, d):
        return m / d if d and d > 0 else 0.0

    total_mass_no_air = sum(m for _, _, m in rows)
    print("\n=== Mix Design Results ===")

    W_MAT, W_DENS, W_MASS, W_VOL, W_PCT = 20, 16, 11, 11, 7
    print(
        f"{'Material':<{W_MAT}}"
        f"{'Density (kg/L)':>{W_DENS}}"
        f"{'Mass (kg)':>{W_MASS}}"
        f"{'Volume (L)':>{W_VOL}}"
        f"{'Mass %':>{W_PCT}}"
    )

    total_mass = 0.0
    total_vol = 0.0
    for name, dens, mass in rows:
        vol = vol_l(mass, dens)
        pct = (100.0 * mass / total_mass_no_air) if total_mass_no_air > 0 else 0.0
        total_mass += mass
        total_vol += vol
        print(
            f"{name:<{W_MAT}}"
            f"{dens:>{W_DENS}.2f}"
            f"{mass:>{W_MASS}.2f}"
            f"{vol:>{W_VOL}.2f}"
            f"{pct:>{W_PCT}.1f}"
        )

    air_vol_L = (air_pct / 100.0) * 1000.0
    if air_pct > 0:
        print(
            f"{'Air':<{W_MAT}}"
            f"{'-':>{W_DENS}}"
            f"{0.0:>{W_MASS}.2f}"
            f"{air_vol_L:>{W_VOL}.2f}"
            f"{'':>{W_PCT}}"
        )
        total_vol += air_vol_L

    avg_density = total_mass / total_vol if total_vol > 0 else 0.0
    print(
        f"{'Total':<{W_MAT}}"
        f"{avg_density:>{W_DENS}.2f}"
        f"{total_mass:>{W_MASS}.2f}"
        f"{total_vol:>{W_VOL}.2f}"
        f"{'':>{W_PCT}}"
    )
    print(f"\nTotal Binder (kg/m³): {btot:.2f}")