import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from lcc.dataset import build_df
from lcc.models import (
    build_models,
    predict_f3_from_wb,
    predict_f7_from_wb,
    predict_f14_from_wb,
    implied_f28_from_wb,
)
from lcc.design import design_mix_from_strengths_min


STRENGTH_MATCH_TOL_MPA = 0.1


st.set_page_config(
    page_title="Low-Carbon Concrete Mix Design Tool",
    page_icon="🧱",
    layout="wide",
)

st.title("Low-Carbon Concrete Mix Design Tool")
st.caption("For research and preliminary mix-design estimation only.")

st.markdown(
"""
### How to use this tool

1. Select the **concrete family**
2. Choose the **required early-age strength age**
3. Enter the **minimum early-age strength**
4. Enter the **minimum 28-day strength**
5. Click **Generate Mix Design**
"""
)

# ---------------------------------------------------------
# Build dataset and models
# ---------------------------------------------------------

@st.cache_data
def load_models():
    df = build_df()
    models = build_models(df)
    return models

models = load_models()


# ---------------------------------------------------------
# Helpers for nice mix design output
# ---------------------------------------------------------

DISPLAY_ORDER = [
    ("inputs.min_early_MPa", "Minimum early-age strength (MPa)"),
    ("inputs.early_age_days", "Early-age requirement (days)"),
    ("inputs.min_28d_MPa", "Minimum 28-day strength (MPa)"),
    ("inputs.binder_family", "Concrete family"),
    ("predicted_parameters.water_binder_ratio", "Water–binder ratio"),
    ("predicted_parameters.water_kg_m3", "Water (kg/m³)"),
    ("predicted_parameters.binder_total_kg_m3", "Total binder (kg/m³)"),
    ("binder_exact.Cement", "Cement (kg/m³)"),
    ("binder_exact.GGBFS", "GGBFS (kg/m³)"),
    ("binder_exact.Fly Ash", "Fly Ash (kg/m³)"),
    ("admixture_split_kg_m3.Plastiment 30", "Plastiment 30 (kg/m³)"),
    ("admixture_split_kg_m3.ECO WR", "ECO WR (kg/m³)"),
    ("admixture_split_kg_m3.Retarder", "Retarder (kg/m³)"),
    ("aggregates_exact.20mm Aggregate", "20mm Aggregate (kg/m³)"),
    ("aggregates_exact.10mm Aggregate", "10mm Aggregate (kg/m³)"),
    ("aggregates_exact.Man Sand", "Manufactured Sand (kg/m³)"),
    ("aggregates_exact.Natural Sand", "Natural Sand (kg/m³)"),
    ("predicted_parameters.fresh_density_target_kg_m3", "Fresh density target (kg/m³)"),
    ("predicted_parameters.air_percent", "Air content (%)"),
    ("embodied_carbon.EC_A1", "Embodied carbon A1 (kgCO₂-e/m³)"),
    ("embodied_carbon.EC_A2", "Embodied carbon A2 (kgCO₂-e/m³)"),
    ("embodied_carbon.EC_A3", "Embodied carbon A3 (kgCO₂-e/m³)"),
    ("embodied_carbon.EC_total", "Embodied carbon total (kgCO₂-e/m³)"),
    ("totals.sum_all_components_kg_m3", "Total mass of all components (kg/m³)"),
]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def flatten_dict(d, parent_key=""):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key))
        else:
            items[new_key] = v
    return items


def format_mix_output(result_dict):
    flat = flatten_dict(result_dict)
    rows = []
    seen = set()

    for key, label in DISPLAY_ORDER:
        if key in flat:
            val = flat[key]
            num = safe_float(val)
            rows.append(
                {
                    "Item": label,
                    "Value": round(num, 3) if num is not None else val,
                }
            )
            seen.add(key)

    for key, val in flat.items():
        if key in seen or key.endswith("wb_override"):
            continue
        num = safe_float(val)
        rows.append(
            {
                "Item": key,
                "Value": round(num, 3) if num is not None else val,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# Feasibility helpers
# ---------------------------------------------------------

def get_allowed_wb_bounds(family_key, early_age_days):
    wb_min = 0.34
    wb_max = 0.75

    if int(early_age_days) == 3:
        wb_caps_3d = {
            "P1": 0.75,
            "F2": 0.70,
            "F4": 0.64,
            "F5": 0.58,
            "S3": 0.62,
            "S5": 0.55,
            "S6": 0.48,
            "T1": 0.52,
            "T2": 0.49,
        }
        wb_max = min(wb_max, wb_caps_3d.get(family_key, wb_max))

    return float(wb_min), float(wb_max)


def get_strength_at_age(models, family_key, age_days, wb):
    if int(age_days) == 3:
        return float(predict_f3_from_wb(models, wb, family_key))
    elif int(age_days) == 7:
        return float(predict_f7_from_wb(models, wb, family_key))
    elif int(age_days) == 14:
        return float(predict_f14_from_wb(models, wb, family_key))
    else:
        return float(implied_f28_from_wb(models, wb, family_key))


def get_strength_range_for_family(models, family_key, age_days, wb_min, wb_max):
    f_at_min_wb = get_strength_at_age(models, family_key, age_days, wb_min)
    f_at_max_wb = get_strength_at_age(models, family_key, age_days, wb_max)
    return (
        min(f_at_min_wb, f_at_max_wb),
        max(f_at_min_wb, f_at_max_wb),
    )


# ---------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------

def build_strength_curve_df(models, families, age_days):
    wb_values = np.linspace(0.34, 0.75, 120)
    rows = []

    for fam in families:
        for wb in wb_values:
            if age_days == 3:
                strength = predict_f3_from_wb(models, wb, fam)
            elif age_days == 7:
                strength = predict_f7_from_wb(models, wb, fam)
            elif age_days == 14:
                strength = predict_f14_from_wb(models, wb, fam)
            else:
                strength = implied_f28_from_wb(models, wb, fam)

            rows.append(
                {
                    "Water/Binder ratio": wb,
                    "Compressive strength": strength,
                    "Binder family": fam,
                    "Age label": f"{age_days} Day",
                }
            )

    return pd.DataFrame(rows)


def build_family_multiage_curve_df(models, family_key, age_list=(3, 7, 14, 28)):
    wb_values = np.linspace(0.34, 0.75, 120)
    rows = []

    for age_days in age_list:
        for wb in wb_values:
            if age_days == 3:
                strength = predict_f3_from_wb(models, wb, family_key)
            elif age_days == 7:
                strength = predict_f7_from_wb(models, wb, family_key)
            elif age_days == 14:
                strength = predict_f14_from_wb(models, wb, family_key)
            else:
                strength = implied_f28_from_wb(models, wb, family_key)

            rows.append(
                {
                    "Water/Binder ratio": wb,
                    "Compressive strength": strength,
                    "Binder family": family_key,
                    "Age label": f"{age_days} Day",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# INPUTS
# ---------------------------------------------------------

st.sidebar.header("Inputs")

FAMILY_OPTIONS = ["P1", "F2", "F4", "F5", "S3", "S5", "S6", "T1", "T2"]

family = st.sidebar.selectbox(
    "Binder family",
    FAMILY_OPTIONS
)

early_age_label = st.sidebar.selectbox(
    "Early-age strength age",
    ["3 Day", "7 Day", "14 Day"]
)

early_age_days = {
    "3 Day": 3,
    "7 Day": 7,
    "14 Day": 14
}[early_age_label]

min_early_strength = st.sidebar.number_input(
    "Minimum early-age strength (MPa)",
    0.0,
    100.0,
    25.0,
)

min_28_strength = st.sidebar.number_input(
    "Minimum 28-day strength (MPa)",
    0.0,
    100.0,
    40.0,
)

generate = st.sidebar.button("Generate Mix Design")


# ---------------------------------------------------------
# RUN MIX DESIGN
# ---------------------------------------------------------

if generate:

    mix = design_mix_from_strengths_min(
        models=models,
        early_min=min_early_strength,
        f28_min=min_28_strength,
        binder_family_key=family,
        early_age_days=early_age_days,
    )

    wb = float(mix["predicted_parameters"]["water_binder_ratio"])

    pred_f3 = float(predict_f3_from_wb(models, wb, family))
    pred_f7 = float(predict_f7_from_wb(models, wb, family))
    pred_f14 = float(predict_f14_from_wb(models, wb, family))
    pred_f28 = float(implied_f28_from_wb(models, wb, family))

    if early_age_days == 3:
        pred_early = pred_f3
    elif early_age_days == 7:
        pred_early = pred_f7
    else:
        pred_early = pred_f14

    pp = mix["predicted_parameters"]
    binder_total = float(pp["binder_total_kg_m3"])
    water = float(pp["water_kg_m3"])
    ec_total = float(mix["embodied_carbon"]["EC_total"])
    result_df = format_mix_output(mix)

    # -----------------------------------------------------
    # Feasibility checks
    # -----------------------------------------------------
    wb_min_allowed, wb_max_allowed = get_allowed_wb_bounds(family, early_age_days)

    early_lo, early_hi = get_strength_range_for_family(
        models=models,
        family_key=family,
        age_days=early_age_days,
        wb_min=wb_min_allowed,
        wb_max=wb_max_allowed,
    )

    d28_lo, d28_hi = get_strength_range_for_family(
        models=models,
        family_key=family,
        age_days=28,
        wb_min=wb_min_allowed,
        wb_max=wb_max_allowed,
    )

    feasibility_warnings = []

    if float(min_early_strength) < early_lo:
        feasibility_warnings.append(
            f"{early_age_days}-day requirement ({float(min_early_strength):.2f} MPa) is below the achievable range "
            f"for {family} ({early_lo:.2f} to {early_hi:.2f} MPa) within allowed w/b limits "
            f"({wb_min_allowed:.2f} to {wb_max_allowed:.2f})."
        )
    elif float(min_early_strength) > early_hi:
        feasibility_warnings.append(
            f"{early_age_days}-day requirement ({float(min_early_strength):.2f} MPa) is above the achievable range "
            f"for {family} ({early_lo:.2f} to {early_hi:.2f} MPa) within allowed w/b limits "
            f"({wb_min_allowed:.2f} to {wb_max_allowed:.2f})."
        )

    if float(min_28_strength) < d28_lo:
        feasibility_warnings.append(
            f"28-day requirement ({float(min_28_strength):.2f} MPa) is below the achievable range "
            f"for {family} ({d28_lo:.2f} to {d28_hi:.2f} MPa) within allowed w/b limits "
            f"({wb_min_allowed:.2f} to {wb_max_allowed:.2f})."
        )
    elif float(min_28_strength) > d28_hi:
        feasibility_warnings.append(
            f"28-day requirement ({float(min_28_strength):.2f} MPa) is above the achievable range "
            f"for {family} ({d28_lo:.2f} to {d28_hi:.2f} MPa) within allowed w/b limits "
            f"({wb_min_allowed:.2f} to {wb_max_allowed:.2f})."
        )

    # -----------------------------------------------------
    # Requirement match warnings
    # -----------------------------------------------------
    match_warnings = []

    early_diff = float(pred_early) - float(min_early_strength)
    d28_diff = float(pred_f28) - float(min_28_strength)

    if abs(early_diff) > STRENGTH_MATCH_TOL_MPA:
        if early_diff > 0:
            match_warnings.append(
                f"{early_age_days}-day predicted strength ({pred_early:.2f} MPa) is "
                f"{early_diff:.2f} MPa above the requirement ({min_early_strength:.2f} MPa)."
            )
        else:
            match_warnings.append(
                f"{early_age_days}-day predicted strength ({pred_early:.2f} MPa) is "
                f"{abs(early_diff):.2f} MPa below the requirement ({min_early_strength:.2f} MPa)."
            )

    if abs(d28_diff) > STRENGTH_MATCH_TOL_MPA:
        if d28_diff > 0:
            match_warnings.append(
                f"28-day predicted strength ({pred_f28:.2f} MPa) is "
                f"{d28_diff:.2f} MPa above the requirement ({min_28_strength:.2f} MPa)."
            )
        else:
            match_warnings.append(
                f"28-day predicted strength ({pred_f28:.2f} MPa) is "
                f"{abs(d28_diff):.2f} MPa below the requirement ({min_28_strength:.2f} MPa)."
            )

    tab1, tab2 = st.tabs(["Generated Mix Design", "Predicted Strengths"])

    with tab1:
        st.header("Generated Mix Design")

        if feasibility_warnings:
            for msg in feasibility_warnings:
                st.warning(msg)
            st.info("Closest feasible mix within the model bounds has been returned.")

        if match_warnings:
            for msg in match_warnings:
                st.warning(msg)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Concrete family", family)
        c2.metric("Estimated w/b ratio", f"{wb:.3f}")
        c3.metric("Water", f"{water:.1f} kg/m³")
        c4.metric("Total binder", f"{binder_total:.1f} kg/m³")

        c5, c6, c7 = st.columns(3)
        c5.metric("28-day requirement", f"≥ {min_28_strength:.1f} MPa")
        c6.metric("Early-age requirement", f"≥ {min_early_strength:.1f} MPa")
        c7.metric("Embodied carbon", f"{ec_total:.1f} kgCO₂-e/m³")

        st.dataframe(result_df, use_container_width=True, hide_index=True)

    with tab2:
        st.header("Predicted Strengths")

        if feasibility_warnings:
            st.info("Displayed strengths correspond to the closest feasible mix returned by the model.")

        if match_warnings:
            for msg in match_warnings:
                st.warning(msg)

        strength_df = pd.DataFrame(
            {
                "Age": [3, 7, 14, 28],
                "Strength (MPa)": [
                    round(pred_f3, 2),
                    round(pred_f7, 2),
                    round(pred_f14, 2),
                    round(pred_f28, 2),
                ],
            }
        )

        st.dataframe(strength_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# GRAPH SECTION
# ---------------------------------------------------------

st.header("Strength vs Water–Binder Ratio")

graph_mode = st.radio(
    "Graph mode",
    [
        "Selected family - all ages",
        "All families - one age",
    ],
    horizontal=True,
)

if graph_mode == "All families - one age":
    graph_age_label = st.selectbox(
        "Graph strength age",
        ["3 Day", "7 Day", "14 Day", "28 Day"],
        index=3
    )

    graph_age_days = {
        "3 Day": 3,
        "7 Day": 7,
        "14 Day": 14,
        "28 Day": 28
    }[graph_age_label]

    curve_df = build_strength_curve_df(
        models,
        FAMILY_OPTIONS,
        graph_age_days
    )

    if graph_age_days == 28:
        chosen_strength = float(min_28_strength)
    elif graph_age_days == early_age_days:
        chosen_strength = float(min_early_strength)
    else:
        chosen_strength = 0.0

    base = alt.Chart(curve_df)

    chart = (
        base
        .mark_line()
        .encode(
            x=alt.X(
                "Water/Binder ratio:Q",
                title="Water/Binder ratio"
            ),
            y=alt.Y(
                "Compressive strength:Q",
                title="Compressive strength (MPa)"
            ),
            color=alt.Color(
                "Binder family:N",
                title="Binder family"
            ),
            tooltip=[
                alt.Tooltip("Binder family:N"),
                alt.Tooltip("Age label:N"),
                alt.Tooltip("Water/Binder ratio:Q", format=".3f"),
                alt.Tooltip("Compressive strength:Q", format=".2f"),
            ],
        )
        .properties(height=450)
        .interactive()
    )

    if chosen_strength > 0:
        rule_df = pd.DataFrame(
            {
                "Target strength": [chosen_strength],
                "Label": [f"Target = {chosen_strength:.1f} MPa"],
            }
        )

        rule = (
            alt.Chart(rule_df)
            .mark_rule(strokeDash=[8, 6], size=2)
            .encode(
                y=alt.Y("Target strength:Q"),
                tooltip=[alt.Tooltip("Label:N")],
            )
        )

        text = (
            alt.Chart(rule_df)
            .mark_text(align="left", dx=8, dy=-6)
            .encode(
                x=alt.value(10),
                y=alt.Y("Target strength:Q"),
                text="Label:N",
            )
        )

        st.altair_chart(chart + rule + text, use_container_width=True)
    else:
        st.altair_chart(chart, use_container_width=True)

else:
    family_curve_df = build_family_multiage_curve_df(
        models=models,
        family_key=family,
        age_list=(3, 7, 14, 28),
    )

    show_early_line = st.checkbox("Show early-age target line", value=True)
    show_28_line = st.checkbox("Show 28-day target line", value=True)

    base = alt.Chart(family_curve_df)

    chart = (
        base
        .mark_line()
        .encode(
            x=alt.X(
                "Water/Binder ratio:Q",
                title="Water/Binder ratio"
            ),
            y=alt.Y(
                "Compressive strength:Q",
                title="Compressive strength (MPa)"
            ),
            color=alt.Color(
                "Age label:N",
                title="Strength age"
            ),
            tooltip=[
                alt.Tooltip("Binder family:N"),
                alt.Tooltip("Age label:N"),
                alt.Tooltip("Water/Binder ratio:Q", format=".3f"),
                alt.Tooltip("Compressive strength:Q", format=".2f"),
            ],
        )
        .properties(height=450)
        .interactive()
    )

    layers = [chart]

    if show_early_line:
        early_rule_df = pd.DataFrame(
            {
                "Target strength": [float(min_early_strength)],
                "Label": [f"Early target = {float(min_early_strength):.1f} MPa"],
            }
        )

        early_rule = (
            alt.Chart(early_rule_df)
            .mark_rule(strokeDash=[8, 6], size=2)
            .encode(
                y=alt.Y("Target strength:Q"),
                tooltip=[alt.Tooltip("Label:N")],
            )
        )

        early_text = (
            alt.Chart(early_rule_df)
            .mark_text(align="left", dx=8, dy=-6)
            .encode(
                x=alt.value(10),
                y=alt.Y("Target strength:Q"),
                text="Label:N",
            )
        )

        layers.extend([early_rule, early_text])

    if show_28_line:
        d28_rule_df = pd.DataFrame(
            {
                "Target strength": [float(min_28_strength)],
                "Label": [f"28-day target = {float(min_28_strength):.1f} MPa"],
            }
        )

        d28_rule = (
            alt.Chart(d28_rule_df)
            .mark_rule(strokeDash=[8, 6], size=2)
            .encode(
                y=alt.Y("Target strength:Q"),
                tooltip=[alt.Tooltip("Label:N")],
            )
        )

        d28_text = (
            alt.Chart(d28_rule_df)
            .mark_text(align="left", dx=8, dy=-6)
            .encode(
                x=alt.value(10),
                y=alt.Y("Target strength:Q"),
                text="Label:N",
            )
        )

        layers.extend([d28_rule, d28_text])

    final_chart = layers[0]
    for layer in layers[1:]:
        final_chart = final_chart + layer

    st.altair_chart(final_chart, use_container_width=True)
    st.caption(f"Showing 3, 7, 14 and 28-day strength curves for selected family: {family}")