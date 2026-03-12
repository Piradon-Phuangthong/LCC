from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .config import BINDER_FAMILIES, USE_FIXED_WATER, WATER_FIXED


@dataclass
class ModelsBundle:
    # 28-day Abram curves
    family_curves: Dict[str, Tuple[float, float]]
    A_g: float
    b_g: float

    # 3-day Abram curves
    family_curves_f3: Dict[str, Tuple[float, float]]
    A3_g: float
    b3_g: float

    # 7-day Abram curves
    family_curves_f7: Dict[str, Tuple[float, float]]
    A7_g: float
    b7_g: float

    # inverse wb from 28d
    wb_inv_models: Dict[str, KNeighborsRegressor]
    wb_ranges: Dict[str, Tuple[float, float]]

    # water models
    family_models_water_3d: Dict[str, KNeighborsRegressor]
    knn_water_global_3d: KNeighborsRegressor

    family_models_water_7d: Dict[str, KNeighborsRegressor]
    knn_water_global_7d: KNeighborsRegressor

    W_MIN: float
    W_MAX: float


def _power_model(x, A, b):
    return A * (x ** b)


def fit_abram_curve_xy(x_vals, y_vals, default_A=100.0, default_b=-1.8):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    if len(x) < 2:
        return float(default_A), float(default_b)

    try:
        from scipy.optimize import curve_fit  # type: ignore

        b0, a0 = np.polyfit(np.log(x), np.log(y), 1)
        A0 = float(np.exp(a0))

        (A, b), _ = curve_fit(
            _power_model,
            x,
            y,
            p0=(A0, b0),
            bounds=([1e-6, -4.5], [1e4, -0.2]),
            maxfev=20000,
        )
        return float(A), float(b)

    except Exception:
        lx, ly = np.log(x), np.log(y)
        b, a = np.polyfit(lx, ly, 1)
        A = float(np.exp(a))
        b = float(min(-0.2, max(-4.5, b)))
        return float(A), float(b)


def fit_abram_curve(df_sub):
    return fit_abram_curve_xy(
        df_sub["Water/Binder"].values,
        df_sub["Strength28d"].values,
        default_A=100.0,
        default_b=-1.8,
    )


def fit_abram_curve_f3(df_sub):
    return fit_abram_curve_xy(
        df_sub["Water/Binder"].values,
        df_sub["Strength3d"].values,
        default_A=55.0,
        default_b=-1.5,
    )


def fit_abram_curve_f7(df_sub):
    return fit_abram_curve_xy(
        df_sub["Water/Binder"].values,
        df_sub["Strength7d"].values,
        default_A=75.0,
        default_b=-1.6,
    )


def build_models(df) -> ModelsBundle:
    # -------------------------
    # Inverse wb models from 28d
    # -------------------------
    wb_inv_models: Dict[str, KNeighborsRegressor] = {}
    wb_ranges: Dict[str, Tuple[float, float]] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Strength28d", "Water/Binder"]].dropna()
        if len(sub) >= 2:
            mdl = KNeighborsRegressor(n_neighbors=min(4, len(sub)), weights="distance")
            mdl.fit(sub[["Strength28d"]].values, sub["Water/Binder"].values)
            wb_inv_models[fam] = mdl
            wb_ranges[fam] = (
                float(sub["Strength28d"].min()),
                float(sub["Strength28d"].max()),
            )

    # -------------------------
    # 28-day Abram curves
    # -------------------------
    family_curves: Dict[str, Tuple[float, float]] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Water/Binder", "Strength28d"]].dropna()
        if len(sub) >= 2:
            family_curves[fam] = fit_abram_curve(sub)
    A_g, b_g = fit_abram_curve(df[["Water/Binder", "Strength28d"]].dropna())

    # -------------------------
    # 3-day Abram curves
    # -------------------------
    family_curves_f3: Dict[str, Tuple[float, float]] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Water/Binder", "Strength3d"]].dropna()
        if len(sub) >= 2:
            family_curves_f3[fam] = fit_abram_curve_f3(sub)
    df_f3 = df[["Water/Binder", "Strength3d"]].dropna()
    A3_g, b3_g = fit_abram_curve_f3(df_f3)

    # -------------------------
    # 7-day Abram curves
    # -------------------------
    family_curves_f7: Dict[str, Tuple[float, float]] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Water/Binder", "Strength7d"]].dropna()
        if len(sub) >= 2:
            family_curves_f7[fam] = fit_abram_curve_f7(sub)
    df_f7 = df[["Water/Binder", "Strength7d"]].dropna()
    A7_g, b7_g = fit_abram_curve_f7(df_f7)

    # -------------------------
    # Water models (3d)
    # -------------------------
    family_models_water_3d: Dict[str, KNeighborsRegressor] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam].dropna(
            subset=["Strength3d", "Strength28d", "Water/Binder", "Free Water"]
        )
        if len(sub) >= 2:
            family_models_water_3d[fam] = KNeighborsRegressor(
                n_neighbors=min(3, len(sub)),
                weights="distance",
            ).fit(
                sub[["Strength3d", "Strength28d", "Water/Binder"]].values,
                sub["Free Water"].values,
            )

    df_w3 = df.dropna(
        subset=["Strength3d", "Strength28d", "Water/Binder", "Free Water"]
    ).copy()
    knn_water_global_3d = KNeighborsRegressor(n_neighbors=min(3, len(df_w3)), weights="distance").fit(
        df_w3[["Strength3d", "Strength28d", "Water/Binder"]].values,
        df_w3["Free Water"].values,
    )

    # -------------------------
    # Water models (7d)
    # -------------------------
    family_models_water_7d: Dict[str, KNeighborsRegressor] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam].dropna(
            subset=["Strength7d", "Strength28d", "Water/Binder", "Free Water"]
        )
        if len(sub) >= 2:
            family_models_water_7d[fam] = KNeighborsRegressor(
                n_neighbors=min(3, len(sub)),
                weights="distance",
            ).fit(
                sub[["Strength7d", "Strength28d", "Water/Binder"]].values,
                sub["Free Water"].values,
            )

    df_w7 = df.dropna(
        subset=["Strength7d", "Strength28d", "Water/Binder", "Free Water"]
    ).copy()
    knn_water_global_7d = KNeighborsRegressor(n_neighbors=min(3, len(df_w7)), weights="distance").fit(
        df_w7[["Strength7d", "Strength28d", "Water/Binder"]].values,
        df_w7["Free Water"].values,
    )

    W_MIN = float(df["Free Water"].min())
    W_MAX = float(df["Free Water"].max())

    return ModelsBundle(
        family_curves=family_curves,
        A_g=A_g,
        b_g=b_g,
        family_curves_f3=family_curves_f3,
        A3_g=A3_g,
        b3_g=b3_g,
        family_curves_f7=family_curves_f7,
        A7_g=A7_g,
        b7_g=b7_g,
        wb_inv_models=wb_inv_models,
        wb_ranges=wb_ranges,
        family_models_water_3d=family_models_water_3d,
        knn_water_global_3d=knn_water_global_3d,
        family_models_water_7d=family_models_water_7d,
        knn_water_global_7d=knn_water_global_7d,
        W_MIN=W_MIN,
        W_MAX=W_MAX,
    )


def _wb_from_curve(models: ModelsBundle, f28: float, fam_key: str) -> float:
    A, b = models.family_curves.get(fam_key, (models.A_g, models.b_g))
    if abs(b) < 1e-9:
        return 0.50
    return float((max(1e-6, f28) / A) ** (1.0 / b))


def _wb_from_knn(models: ModelsBundle, f28: float, fam_key: str) -> Optional[float]:
    mdl = models.wb_inv_models.get(fam_key)
    if mdl is None:
        return None
    fmin, fmax = models.wb_ranges[fam_key]
    f_in = min(max(float(f28), fmin), fmax)
    return float(mdl.predict([[f_in]])[0])


def predict_wb_from_f28_curve(models: ModelsBundle, f28: float, fam_key: str) -> float:
    wb_curve = _wb_from_curve(models, f28, fam_key)
    wb_knn = _wb_from_knn(models, f28, fam_key)

    if wb_knn is None:
        wb = wb_curve
    else:
        f = float(f28)
        w_data = 0.7 if f <= 40 else (0.4 if f >= 60 else 0.7 - (0.3 * (f - 40) / 20.0))
        wb = w_data * wb_knn + (1.0 - w_data) * wb_curve

    wb = min(wb, wb_curve)
    return max(0.34, min(0.75, wb))


def _wb_from_age_curve(
    target_strength: float,
    A: float,
    b: float,
    wb_min: float = 0.34,
    wb_max: float = 0.75,
) -> float:
    if abs(b) < 1e-9 or A <= 0.0 or target_strength <= 0.0:
        return 0.50
    wb = float((max(1e-6, target_strength) / A) ** (1.0 / b))
    return max(wb_min, min(wb_max, wb))


def implied_f28_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    A, b = models.family_curves.get(fam_key, (models.A_g, models.b_g))
    return float(A * (float(wb) ** b))


def predict_f3_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    A, b = models.family_curves_f3.get(fam_key, (models.A3_g, models.b3_g))
    return float(A * (float(wb) ** b))


def predict_f7_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    A, b = models.family_curves_f7.get(fam_key, (models.A7_g, models.b7_g))
    return float(A * (float(wb) ** b))


def predict_wb_from_f3_anchor(
    models: ModelsBundle,
    f3_target: float,
    fam_key: str,
    wb_min: float = 0.34,
    wb_max: float = 0.75,
    n_grid: int = 500,
) -> float:
    A, b = models.family_curves_f3.get(fam_key, (models.A3_g, models.b3_g))
    return _wb_from_age_curve(
        target_strength=float(f3_target),
        A=A,
        b=b,
        wb_min=wb_min,
        wb_max=wb_max,
    )


def predict_wb_from_f7_anchor(
    models: ModelsBundle,
    f7_target: float,
    fam_key: str,
    wb_min: float = 0.34,
    wb_max: float = 0.75,
    n_grid: int = 500,
) -> float:
    A, b = models.family_curves_f7.get(fam_key, (models.A7_g, models.b7_g))
    return _wb_from_age_curve(
        target_strength=float(f7_target),
        A=A,
        b=b,
        wb_min=wb_min,
        wb_max=wb_max,
    )


def predict_f14_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    f28 = float(implied_f28_from_wb(models, wb, fam_key))
    f7 = float(predict_f7_from_wb(models, wb, fam_key))

    if f28 <= 0.0:
        return 0.0

    r = f7 / f28
    r = max(0.01, min(0.999, float(r)))

    denom = np.log(7.0 / 28.0)
    n = float(np.log(r) / denom) if denom != 0.0 else 1.0

    f14 = float(f28 * ((14.0 / 28.0) ** n))

    lo = min(f7, f28)
    hi = max(f7, f28)
    return max(lo, min(hi, f14))


def get_water(
    models: ModelsBundle,
    early_strength: float,
    f28: float,
    wb: float,
    fam_key: str,
    early_age_days: int = 3,
) -> float:
    if USE_FIXED_WATER:
        return float(WATER_FIXED)

    if int(early_age_days) in (7, 14):
        mdl_w = models.family_models_water_7d.get(fam_key, models.knn_water_global_7d)
        w = float(mdl_w.predict([[float(early_strength), float(f28), float(wb)]])[0])
    else:
        mdl_w = models.family_models_water_3d.get(fam_key, models.knn_water_global_3d)
        w = float(mdl_w.predict([[float(early_strength), float(f28), float(wb)]])[0])

    return max(models.W_MIN, min(models.W_MAX, w))