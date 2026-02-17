# app.py
# Low-Carbon Concrete Mix Design Tool — Streamlit UI
# Run:
#   pip install streamlit scikit-learn numpy pandas scipy
#   streamlit run app.py
#
# IMPORTANT:
# - This UI is aligned with your current CLI pipeline (LCC.py) and the lcc/ package.
# - It does NOT re-implement models/dataset/EC logic. It calls:
#     build_df() -> build_models() -> design_mix_from_strengths_min()
#   and uses the same reporting functions for predicted strengths.

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from lcc.config import BINDER_FAMILIES, USE_FIXED_WATER, WATER_FIXED
from lcc.dataset import build_df
from lcc.models import (
    build_models,
    implied_f28_from_wb,
    predict_f3_from_wb,
    predict_f7_from_wb,
    predict_wb_from_f28_curve,
)
from lcc.design import design_mix_from_strengths_min


# =============================================================================
# Page configuration (professional / neutral)
# =============================================================================
st.set_page_config(
    page_title="Low-Carbon Concrete Mix Design Tool",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# Styling (lightweight, professional)
# =============================================================================
st.markdown(
    """
<style>
/* overall spacing */
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; }

/* headings */
h1, h2, h3 { letter-spacing: -0.2px; }

/* sidebar refinements */
section[data-testid="stSidebar"] .block-container { padding-top: 1.1rem; }

/* cards (metrics + info) */
div[data-testid="stMetric"] { background: rgba(250,250,250,0.65); border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px; padding: 12px 14px; }

/* expander */
div[data-testid="stExpander"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(0,0,0,0.06); }

/* dataframe border */
div[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid rgba(0,0,0,0.06); }

/* subtle caption */
.small-muted { color: rgba(0,0,0,0.55); font-size: 0.92rem; }

/* badges */
.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 0.85rem;
  border: 1px solid rgba(0,0,0,0.10);
  background: rgba(0,0,0,0.03);
}
.badge-good { background: rgba(0, 200, 83, 0.10); border-color: rgba(0, 200, 83, 0.25); }
.badge-warn { background: rgba(255, 193, 7, 0.12); border-color: rgba(255, 193, 7, 0.25); }
.badge-info { background: rgba(33, 150, 243, 0.10); border-color: rgba(33, 150, 243, 0.25); }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# Presentation constants (for tables only)
# =============================================================================
DENSITY_KG_PER_L = {
    "Water": 1.00,
    "Cement": 3.11,
    "Fly Ash": 2.35,
    "GGBFS": 2.89,
    "20mm Aggregate": 2.61,
    "10mm Aggregate": 2.71,
    "Man Sand": 2.64,
    "Natural Sand": 2.65,
    "Plastiment 30": 1.05,
    "ECO WR": 1.07,
    "Retarder": 1.05,
}

# Stable order (✅ includes F5)
FAMILY_ORDER = ["P1", "F2", "F4", "F5", "S3", "S5", "S6", "T1", "T2", "T3"]


# =============================================================================
# Cached data/model build (fast reruns)
# =============================================================================
@st.cache_data(show_spinner=False)
def _load_df() -> pd.DataFrame:
    return build_df()


@st.cache_resource(show_spinner=False)
def _load_models(df: pd.DataFrame):
    return build_models(df)


# =============================================================================
# Header
# =============================================================================
left_h, right_h = st.columns([0.72, 0.28], vertical_alignment="center")
with left_h:
    st.title("Low-Carbon Concrete Mix Design Tool")
    st.markdown(
        "<div class='small-muted'>Generate indicative mix proportions from minimum strength targets and a binder family. "
        "Embodied carbon is reported using the report-aligned A1–A3 method implemented in the LCC codebase.</div>",
        unsafe_allow_html=True,
    )

with right_h:
    if USE_FIXED_WATER:
        st.markdown(
            f"<div class='badge badge-good'>VALIDATION MODE • Water fixed at {float(WATER_FIXED):.0f} kg/m³</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<div class='badge badge-info'>DESIGN MODE • Water predicted by model</div>", unsafe_allow_html=True)

with st.expander("Method and limitations", expanded=False):
    st.markdown(
        """
**Purpose**  
Research prototype to support early-stage mix exploration and consistent reporting.

**What the app uses (same as the CLI)**  
- Dataset + binder family labelling from `lcc.dataset.build_df()`  
- Model fitting from `lcc.models.build_models()`  
- Mix design + embodied carbon from `lcc.design.design_mix_from_strengths_min()`  

**Notes / limitations**  
- Outputs are indicative and depend on the representativeness of the dataset and assumptions in the report method.  
- Results should be verified against project specifications, standards, and trial mixes.
"""
    )

st.divider()

# =============================================================================
# Load df + models
# =============================================================================
with st.spinner("Loading dataset and models..."):
    df = _load_df()
    models = _load_models(df)

# =============================================================================
# Sidebar inputs (aligned to LCC.py)
# =============================================================================
with st.sidebar:
    st.header("Inputs")

    st.caption("These inputs map directly to the CLI pipeline (LCC.py).")

    st.subheader("Binder family")
    fam_keys = list(BINDER_FAMILIES.keys())

    # Sort dropdown using FAMILY_ORDER, but keep any extra keys at end
    fam_keys_sorted = sorted(
        fam_keys, key=lambda k: FAMILY_ORDER.index(k) if k in FAMILY_ORDER else 999
    )

    default_idx = fam_keys_sorted.index("S5") if "S5" in fam_keys_sorted else 0
    fam = st.selectbox("Select binder family", fam_keys_sorted, index=default_idx)

    st.subheader("Strength targets")
    early_age_days = st.radio(
        "Early-age strength age (days)", options=[3, 7], horizontal=True, index=0
    )

    colA, colB = st.columns(2)
    with colA:
        # Keep your original behaviour but make it sensible:
        # - 3-day typically lower than 7-day; defaults are just UI convenience.
        early_default = 25.0 if early_age_days == 3 else 35.0
        early_min = st.number_input(
            f"Minimum {early_age_days}-day strength (MPa)",
            min_value=0.0,
            max_value=120.0,
            value=float(early_default),
            step=0.5,
        )
    with colB:
        f28_min = st.number_input(
            "Minimum 28-day strength (MPa)",
            min_value=0.0,
            max_value=120.0,
            value=50.0,
            step=0.5,
        )

    st.subheader("Water/binder ratio (optional override)")
    suggested_wb_sidebar = float(
        predict_wb_from_f28_curve(models, float(f28_min), str(fam).upper())
    )
    use_wb_override = st.checkbox("Override w/b", value=False)

    wb_override_val: Optional[float] = None
    if use_wb_override:
        wb_override_val = st.number_input(
            "w/b override value",
            min_value=0.20,
            max_value=1.20,
            value=float(round(suggested_wb_sidebar, 3)),
            step=0.01,
            help="If enabled, the app will use this w/b instead of the suggested value.",
        )

    st.caption(f"Suggested w/b (from model): **{suggested_wb_sidebar:.3f}**")

    st.divider()
    run_btn = st.button("Run mix design", type="primary", use_container_width=True)

# =============================================================================
# Helper tables (presentation only)
# =============================================================================
def binder_family_table() -> pd.DataFrame:
    rows = []
    for k, v in BINDER_FAMILIES.items():
        slag = 100.0 * float(v.get("GGBFS", 0.0))
        fly = 100.0 * float(v.get("Fly Ash", 0.0))
        cem = max(0.0, 100.0 - slag - fly)
        rows.append((k, cem, slag, fly))

    # ✅ Updated: include F5 in order
    rows.sort(key=lambda r: FAMILY_ORDER.index(r[0]) if r[0] in FAMILY_ORDER else 999)
    df_out = pd.DataFrame(rows, columns=["Family", "Cement (%)", "GGBFS (%)", "Fly Ash (%)"])
    return df_out


def _materials_table_from_out(out: Dict[str, Any]) -> pd.DataFrame:
    pp = out["predicted_parameters"]
    w = float(pp["water_kg_m3"])
    b = out["binder_exact"]
    a = out["aggregates_exact"]
    adm = out.get("admixture_split_kg_m3", {})

    rows = [
        ("Water", w),
        ("Cement", float(b.get("Cement", 0.0))),
        ("GGBFS", float(b.get("GGBFS", 0.0))),
        ("Fly Ash", float(b.get("Fly Ash", 0.0))),
        ("20mm Aggregate", float(a.get("20mm Aggregate", 0.0))),
        ("10mm Aggregate", float(a.get("10mm Aggregate", 0.0))),
        ("Man Sand", float(a.get("Man Sand", 0.0))),
        ("Natural Sand", float(a.get("Natural Sand", 0.0))),
        ("Plastiment 30", float(adm.get("Plastiment 30", 0.0))),
        ("ECO WR", float(adm.get("ECO WR", 0.0))),
        ("Retarder", float(adm.get("Retarder", 0.0))),
    ]

    df_rows = []
    total_mass = 0.0
    total_vol_L = 0.0

    for name, mass in rows:
        dens = DENSITY_KG_PER_L.get(name)
        vol = (mass / dens) if (dens and dens > 0) else 0.0
        df_rows.append([name, dens, mass, vol])
        total_mass += mass
        total_vol_L += vol

    air_pct = float(pp.get("air_percent", 1.9))
    air_vol_L = (air_pct / 100.0) * 1000.0
    df_rows.append(["Air (assumed)", np.nan, 0.0, air_vol_L])
    total_vol_L += air_vol_L

    df_out = pd.DataFrame(df_rows, columns=["Material", "Density (kg/L)", "Mass (kg/m³)", "Volume (L/m³)"])
    df_out["Mass share (%)"] = np.where(total_mass > 0, 100.0 * df_out["Mass (kg/m³)"] / total_mass, 0.0)

    def fmt(x, nd=2):
        return "" if pd.isna(x) else f"{x:.{nd}f}"

    df_out["Density (kg/L)"] = df_out["Density (kg/L)"].map(lambda x: fmt(x, 2))
    df_out["Mass (kg/m³)"] = df_out["Mass (kg/m³)"].map(lambda x: f"{x:.2f}")
    df_out["Volume (L/m³)"] = df_out["Volume (L/m³)"].map(lambda x: f"{x:.2f}")
    df_out["Mass share (%)"] = df_out["Mass share (%)"].map(lambda x: f"{x:.2f}")

    avg_density = (total_mass / total_vol_L) if total_vol_L else 0.0
    total_row = pd.DataFrame(
        [["Total (average density)", f"{avg_density:.2f}", f"{total_mass:.2f}", f"{total_vol_L:.2f}", ""]],
        columns=df_out.columns,
    )
    return pd.concat([df_out, total_row], ignore_index=True)


def _summary_payload(out: Dict[str, Any]) -> pd.DataFrame:
    pp = out["predicted_parameters"]
    ec = out["embodied_carbon"]
    b = out["binder_exact"]
    a = out["aggregates_exact"]
    adm = out.get("admixture_split_kg_m3", {})

    row = {
        "binder_family": out["inputs"]["binder_family"],
        "early_age_days": out["inputs"]["early_age_days"],
        "min_early_MPa": out["inputs"]["min_early_MPa"],
        "min_28d_MPa": out["inputs"]["min_28d_MPa"],
        "water_binder_ratio": pp["water_binder_ratio"],
        "water_kg_m3": pp["water_kg_m3"],
        "binder_total_kg_m3": pp["binder_total_kg_m3"],
        "cement_kg_m3": float(b.get("Cement", 0.0)),
        "ggbfs_kg_m3": float(b.get("GGBFS", 0.0)),
        "flyash_kg_m3": float(b.get("Fly Ash", 0.0)),
        "agg_20mm_kg_m3": float(a.get("20mm Aggregate", 0.0)),
        "agg_10mm_kg_m3": float(a.get("10mm Aggregate", 0.0)),
        "sand_man_kg_m3": float(a.get("Man Sand", 0.0)),
        "sand_nat_kg_m3": float(a.get("Natural Sand", 0.0)),
        "plastiment30_kg_m3": float(adm.get("Plastiment 30", 0.0)),
        "eco_wr_kg_m3": float(adm.get("ECO WR", 0.0)),
        "retarder_kg_m3": float(adm.get("Retarder", 0.0)),
        "air_percent": pp.get("air_percent", np.nan),
        "fresh_density_target_kg_m3": pp.get("fresh_density_target_kg_m3", np.nan),
        "EC_A1_kgCO2e_m3": ec["EC_A1"],
        "EC_A2_kgCO2e_m3": ec["EC_A2"],
        "EC_A3_kgCO2e_m3": ec["EC_A3"],
        "EC_total_kgCO2e_m3": ec["EC_total"],
    }
    return pd.DataFrame([row])


# =============================================================================
# Main layout
# =============================================================================
left, right = st.columns([1.05, 0.95])

with left:
    st.subheader("Binder family options")
    st.dataframe(binder_family_table(), use_container_width=True, hide_index=True)
    st.caption("Binder families are defined as mass fractions of total binder (as per the LCC configuration).")

with right:
    st.subheader("Workflow")
    st.markdown(
        """
1. Select a binder family.  
2. Select early-age strength age (3 or 7 days) and enter minimum strength targets.  
3. (Optional) Override the suggested water/binder ratio.  
4. Run the mix design to view quantities and embodied carbon (A1–A3).
"""
    )
    st.info("The UI calls the same functions used by `LCC.py`. Results should match the CLI for the same inputs.")

st.divider()

# =============================================================================
# Run + render results
# =============================================================================
if run_btn:
    fam_key = str(fam).upper()

    out = design_mix_from_strengths_min(
        models=models,
        early_min=float(early_min),
        f28_min=float(f28_min),
        binder_family_key=fam_key,
        early_age_days=int(early_age_days),
        wb_override=wb_override_val if use_wb_override else None,
    )

    pp = out["predicted_parameters"]
    ec = out["embodied_carbon"]

    wb_used = float(pp["water_binder_ratio"])
    water_used = float(pp["water_kg_m3"])
    binder_total = float(pp["binder_total_kg_m3"])

    # Reporting-only strengths
    f3_pred = float(predict_f3_from_wb(models, wb_used, fam_key))
    f7_pred = float(predict_f7_from_wb(models, wb_used, fam_key))
    f28_pred = float(implied_f28_from_wb(models, wb_used, fam_key))

    early_pred = f7_pred if int(early_age_days) == 7 else f3_pred
    early_ok = (early_pred + 1e-9) >= float(early_min)
    f28_ok = (f28_pred + 1e-9) >= float(f28_min)

    st.subheader("Results")

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Binder family", out["inputs"]["binder_family"])
    c2.metric("Water/binder ratio", f"{wb_used:.3f}" + (" (override)" if use_wb_override else ""))
    c3.metric("Water content (kg/m³)", f"{water_used:.1f}" + (" (fixed)" if USE_FIXED_WATER else " (estimated)"))
    c4.metric("Binder total (kg/m³)", f"{binder_total:.1f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Fresh density (kg/m³)", f"{float(pp['fresh_density_target_kg_m3']):.1f}")
    c6.metric("Air (assumed, %)", f"{float(pp.get('air_percent', 1.9)):.1f}")
    c7.metric("Embodied carbon (kg CO₂e/m³)", f"{float(ec['EC_total']):.1f}")
    c8.metric("A1 / A2 / A3", f"{ec['EC_A1']:.1f} / {ec['EC_A2']:.1f} / {ec['EC_A3']:.1f}")

    st.markdown(
        f"<div class='badge {'badge-good' if (early_ok and f28_ok) else 'badge-warn'}'>"
        + ("Targets met (reporting-only estimates)." if (early_ok and f28_ok) else "One or more targets not met (reporting-only estimates).")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Strength checks
    st.subheader("Strength checks (reporting only)")

    colL, colR = st.columns(2)
    with colL:
        st.markdown("**Predicted strengths at selected w/b**")
        st.write(f"- 3-day strength: **{f3_pred:.1f} MPa**")
        st.write(f"- 7-day strength: **{f7_pred:.1f} MPa**")
        st.write(f"- 28-day strength (curve-implied): **{f28_pred:.1f} MPa**")

    with colR:
        st.markdown("**Targets**")
        st.write(f"- Early-age target ({early_age_days} days): **≥ {float(early_min):.1f} MPa**")
        st.write(f"- 28-day target: **≥ {float(f28_min):.1f} MPa**")

        if early_ok and f28_ok:
            st.success("Targets are met based on the reporting-only strength estimates.")
        else:
            msg = []
            if not early_ok:
                msg.append(f"Early-age target not met (predicted {early_pred:.1f} MPa).")
            if not f28_ok:
                msg.append(f"28-day target not met (predicted {f28_pred:.1f} MPa).")
            st.warning(" ".join(msg))
            st.caption(
                "Tip: Reduce w/b (use override) and re-run, or review the underlying curve/data fit for that binder family."
            )

    st.divider()

    # Detailed outputs
    tab1, tab2, tab3, tab4 = st.tabs(["Materials", "Binder & aggregates", "Embodied carbon", "Export"])

    with tab1:
        st.write("All quantities are reported per **1 m³** basis.")
        st.dataframe(_materials_table_from_out(out), use_container_width=True, hide_index=True)

    with tab2:
        b = out["binder_exact"]
        a = out["aggregates_exact"]
        adm = out.get("admixture_split_kg_m3", {})

        colA, colB = st.columns(2)

        with colA:
            st.markdown("**Binder split**")
            binder_df = pd.DataFrame(
                {
                    "Component": ["Cement", "GGBFS", "Fly Ash"],
                    "kg/m³": [float(b.get("Cement", 0.0)), float(b.get("GGBFS", 0.0)), float(b.get("Fly Ash", 0.0))],
                }
            )
            binder_df["% of binder"] = 100.0 * binder_df["kg/m³"] / max(1e-9, binder_total)
            binder_df["kg/m³"] = binder_df["kg/m³"].map(lambda x: f"{x:.2f}")
            binder_df["% of binder"] = binder_df["% of binder"].map(lambda x: f"{x:.2f}")
            st.dataframe(binder_df, use_container_width=True, hide_index=True)

            st.markdown("**Admixtures**")
            adm_df = pd.DataFrame(
                {
                    "Admixture": ["Plastiment 30", "ECO WR", "Retarder"],
                    "kg/m³": [
                        float(adm.get("Plastiment 30", 0.0)),
                        float(adm.get("ECO WR", 0.0)),
                        float(adm.get("Retarder", 0.0)),
                    ],
                }
            )
            adm_df["kg/m³"] = adm_df["kg/m³"].map(lambda x: f"{x:.3f}")
            st.dataframe(adm_df, use_container_width=True, hide_index=True)

        with colB:
            st.markdown("**Aggregates**")
            aggs_df = pd.DataFrame(
                {
                    "Aggregate": ["20mm Aggregate", "10mm Aggregate", "Man Sand", "Natural Sand"],
                    "kg/m³": [
                        float(a.get("20mm Aggregate", 0.0)),
                        float(a.get("10mm Aggregate", 0.0)),
                        float(a.get("Man Sand", 0.0)),
                        float(a.get("Natural Sand", 0.0)),
                    ],
                }
            )
            aggs_df["kg/m³"] = aggs_df["kg/m³"].map(lambda x: f"{x:.2f}")
            st.dataframe(aggs_df, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown(
            """
**Embodied carbon reporting (A1–A3)**  
- **A1**: materials  
- **A2**: transport (truck + sea) using fixed distances and emission factors  
- **A3**: manufacturing (constant per m³)  
"""
        )
        ec_detail = pd.DataFrame(
            [
                {
                    "A1 (kg CO₂e/m³)": float(ec["EC_A1"]),
                    "A2 (kg CO₂e/m³)": float(ec["EC_A2"]),
                    "A3 (kg CO₂e/m³)": float(ec["EC_A3"]),
                    "Total (kg CO₂e/m³)": float(ec["EC_total"]),
                }
            ]
        ).round(3)
        st.dataframe(ec_detail, use_container_width=True, hide_index=True)

    with tab4:
        st.markdown("**Download outputs**")

        payload = _summary_payload(out)
        st.download_button(
            label="Download results (CSV)",
            data=payload.to_csv(index=False).encode("utf-8"),
            file_name="mix_design_results.csv",
            mime="text/csv",
        )

        json_bytes = json.dumps(out, indent=2).encode("utf-8")
        st.download_button(
            label="Download results (JSON)",
            data=json_bytes,
            file_name="mix_design_results.json",
            mime="application/json",
        )

        with st.expander("Raw output preview", expanded=False):
            st.json(out)

st.caption(
    "Prototype tool for research and reporting support. "
    "For procurement and approvals, verify assumptions, inputs, and results against project specifications and standards."
)
