# lcc/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from .config import BINDER_FAMILIES, USE_FIXED_WATER, WATER_FIXED


@dataclass
class ModelsBundle:
    # curves
    family_curves: Dict[str, Tuple[float, float]]
    A_g: float
    b_g: float

    # inverse wb
    wb_inv_models: Dict[str, KNeighborsRegressor]
    wb_ranges: Dict[str, Tuple[float, float]]

    # f3 / f7 models
    family_models_f3: Dict[str, KNeighborsRegressor]
    knn_f3_global: KNeighborsRegressor

    family_models_f7: Dict[str, KNeighborsRegressor]
    knn_f7_global: KNeighborsRegressor

    # water models
    family_models_water_3d: Dict[str, KNeighborsRegressor]
    knn_water_global_3d: KNeighborsRegressor

    family_models_water_7d: Dict[str, KNeighborsRegressor]
    knn_water_global_7d: KNeighborsRegressor

    W_MIN: float
    W_MAX: float


def _power_model(x, A, b):
    return A * (x ** b)


def fit_abram_curve(df_sub):
    x = df_sub["Water/Binder"].values.astype(float)
    y = df_sub["Strength28d"].values.astype(float)
    m = (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if len(x) < 2:
        return 100.0, -1.8
    try:
        from scipy.optimize import curve_fit  # type: ignore

        b0, a0 = np.polyfit(np.log(x), np.log(y), 1)
        A0 = float(np.exp(a0))
        (A, b), _ = curve_fit(
            _power_model,
            x,
            y,
            p0=(A0, b0),
            bounds=([1e-6, -4.0], [1e3, -0.6]),
        )
        return float(A), float(b)
    except Exception:
        lx, ly = np.log(x), np.log(y)
        b, a = np.polyfit(lx, ly, 1)
        A = float(np.exp(a))
        if b < -2.1:
            b = -1.9
        return float(A), float(b)


def build_models(df) -> ModelsBundle:
    # -------------------------
    # Inverse wb models
    # -------------------------
    wb_inv_models: Dict[str, KNeighborsRegressor] = {}
    wb_ranges: Dict[str, Tuple[float, float]] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Strength28d", "Water/Binder"]].dropna()
        if len(sub) >= 2:
            mdl = KNeighborsRegressor(n_neighbors=min(4, len(sub)), weights="distance")
            mdl.fit(sub[["Strength28d"]].values, sub["Water/Binder"].values)
            wb_inv_models[fam] = mdl
            wb_ranges[fam] = (float(sub["Strength28d"].min()), float(sub["Strength28d"].max()))

    # -------------------------
    # Abram-style curves
    # -------------------------
    family_curves: Dict[str, Tuple[float, float]] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam]
        if len(sub) >= 2:
            family_curves[fam] = fit_abram_curve(sub)
    A_g, b_g = fit_abram_curve(df)

    # -------------------------
    # f3 (family + global fallback)
    # IMPORTANT: drop NaNs for Strength3d before fitting global model
    # -------------------------
    family_models_f3: Dict[str, KNeighborsRegressor] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Water/Binder", "Strength3d"]].dropna()
        if len(sub) >= 2:
            family_models_f3[fam] = KNeighborsRegressor(
                n_neighbors=min(4, len(sub)), weights="distance"
            ).fit(sub[["Water/Binder"]].values, sub["Strength3d"].values)

    df_f3 = df.dropna(subset=["Strength3d", "Water/Binder", "_slag_frac", "_fly_frac"]).copy()
    knn_f3_global = KNeighborsRegressor(n_neighbors=4, weights="distance").fit(
        df_f3[["Water/Binder", "_slag_frac", "_fly_frac"]].values,
        df_f3["Strength3d"].values,
    )

    # -------------------------
    # f7 (family + global fallback)
    # -------------------------
    family_models_f7: Dict[str, KNeighborsRegressor] = {}
    for fam in BINDER_FAMILIES.keys():
        sub = df[df["_family"] == fam][["Water/Binder", "Strength7d"]].dropna()
        if len(sub) >= 2:
            family_models_f7[fam] = KNeighborsRegressor(
                n_neighbors=min(4, len(sub)), weights="distance"
            ).fit(sub[["Water/Binder"]].values, sub["Strength7d"].values)

    df_f7 = df.dropna(subset=["Strength7d", "Water/Binder", "_slag_frac", "_fly_frac"]).copy()
    knn_f7_global = KNeighborsRegressor(n_neighbors=4, weights="distance").fit(
        df_f7[["Water/Binder", "_slag_frac", "_fly_frac"]].values,
        df_f7["Strength7d"].values,
    )

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
                n_neighbors=3, weights="distance"
            ).fit(
                sub[["Strength3d", "Strength28d", "Water/Binder"]].values,
                sub["Free Water"].values,
            )

    df_w3 = df.dropna(subset=["Strength3d", "Strength28d", "Water/Binder", "Free Water"]).copy()
    knn_water_global_3d = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(
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
                n_neighbors=3, weights="distance"
            ).fit(
                sub[["Strength7d", "Strength28d", "Water/Binder"]].values,
                sub["Free Water"].values,
            )

    df_w7 = df.dropna(subset=["Strength7d", "Strength28d", "Water/Binder", "Free Water"]).copy()
    knn_water_global_7d = KNeighborsRegressor(n_neighbors=3, weights="distance").fit(
        df_w7[["Strength7d", "Strength28d", "Water/Binder"]].values,
        df_w7["Free Water"].values,
    )

    # NOTE: keep W_MIN/W_MAX based on full df water range (NaNs in strengths don't matter)
    W_MIN, W_MAX = float(df["Free Water"].min()), float(df["Free Water"].max())

    return ModelsBundle(
        family_curves=family_curves,
        A_g=A_g,
        b_g=b_g,
        wb_inv_models=wb_inv_models,
        wb_ranges=wb_ranges,
        family_models_f3=family_models_f3,
        knn_f3_global=knn_f3_global,
        family_models_f7=family_models_f7,
        knn_f7_global=knn_f7_global,
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

    # ensure curve-implied f28 >= target f28
    wb = min(wb, wb_curve)

    return max(0.34, min(0.75, wb))


def implied_f28_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    A, b = models.family_curves.get(fam_key, (models.A_g, models.b_g))
    return float(A * (wb ** b))


def predict_f3_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    mdl = models.family_models_f3.get(fam_key)
    if mdl is not None:
        return float(mdl.predict([[wb]])[0])
    fam = BINDER_FAMILIES.get(fam_key, {})
    s = float(fam.get("GGBFS", 0.0))
    f = float(fam.get("Fly Ash", 0.0))
    return float(models.knn_f3_global.predict([[wb, s, f]])[0])

# -------------------------------------------------------------------------
# NEW: 3-day anchor inversion (used only when early_age_days == 3)
# -------------------------------------------------------------------------
def predict_wb_from_f3_anchor(
    models: ModelsBundle,
    f3_target: float,
    fam_key: str,
    wb_min: float = 0.34,
    wb_max: float = 0.75,
    n_grid: int = 500,
) -> float:
    """
    Invert the f3 model to obtain w/b that matches the 3-day target.
    Uses bounded 1D grid search over a realistic w/b range.
    """

    wb_grid = np.linspace(wb_min, wb_max, n_grid)

    # Predict f3 over the grid
    f3_preds = np.array([
        predict_f3_from_wb(models, wb, fam_key)
        for wb in wb_grid
    ])

    # Find wb that minimizes absolute error to target
    idx = np.argmin(np.abs(f3_preds - f3_target))
    return float(wb_grid[idx])


def predict_f7_from_wb(models: ModelsBundle, wb: float, fam_key: str) -> float:
    mdl = models.family_models_f7.get(fam_key)
    if mdl is not None:
        return float(mdl.predict([[wb]])[0])
    fam = BINDER_FAMILIES.get(fam_key, {})
    s = float(fam.get("GGBFS", 0.0))
    f = float(fam.get("Fly Ash", 0.0))
    return float(models.knn_f7_global.predict([[wb, s, f]])[0])


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

    if int(early_age_days) == 7:
        mdl_w = models.family_models_water_7d.get(fam_key, models.knn_water_global_7d)
        w = float(mdl_w.predict([[float(early_strength), float(f28), float(wb)]])[0])
    else:
        mdl_w = models.family_models_water_3d.get(fam_key, models.knn_water_global_3d)
        w = float(mdl_w.predict([[float(early_strength), float(f28), float(wb)]])[0])

    return max(models.W_MIN, min(models.W_MAX, w))
