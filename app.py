import streamlit as st
import pandas as pd

from lcc.dataset import build_df
from lcc.models import (
    build_models,
    predict_f3_from_wb,
    predict_f7_from_wb,
    predict_f14_from_wb,
    implied_f28_from_wb,
    predict_wb_from_f28_curve,
)
from lcc.design import design_mix_from_strengths_min


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
3. Enter the **minimum required early-age strength**
4. Enter the **minimum required 28-day strength**
5. Click **Generate Mix Design**

The tool will estimate:
- water–binder ratio
- 3-day, 7-day, 14-day, and 28-day strengths
- mix quantities
- embodied carbon (A1 + A2 + A3)
"""
)

st.divider()


@st.cache_resource
def load_models():
    df = build_df()
    models = build_models(df)
    return df, models


df, models = load_models()

FAMILY_OPTIONS = ["P1", "F2", "F4", "F5", "S3", "S5", "S6", "T1", "T2"]
AGE_MAP = {"3 Day": 3, "7 Day": 7, "14 Day": 14}

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


st.sidebar.header("Design Inputs")

family = st.sidebar.selectbox("Concrete family", FAMILY_OPTIONS, index=0)
age_option = st.sidebar.selectbox(
    "Required early-age strength",
    list(AGE_MAP.keys()),
    index=1,
)
early_strength = st.sidebar.number_input(
    "Minimum required early-age strength (MPa)",
    min_value=1.0,
    max_value=100.0,
    value=30.0,
    step=1.0,
)
f28_min = st.sidebar.number_input(
    "Minimum required 28-day strength (MPa)",
    min_value=1.0,
    max_value=120.0,
    value=50.0,
    step=1.0,
)

with st.sidebar.expander("Optional settings"):
    use_wb_override = st.checkbox("Override water–binder ratio")
    wb_suggested = predict_wb_from_f28_curve(models, float(f28_min), family)
    wb_override = None
    if use_wb_override:
        wb_override = st.number_input(
            "Water–binder ratio override",
            min_value=0.20,
            max_value=1.20,
            value=float(round(wb_suggested, 3)),
            step=0.01,
            format="%.3f",
        )
    st.caption(f"Suggested w/b from 28-day model: {wb_suggested:.3f}")

run_design = st.sidebar.button("Generate Mix Design", type="primary")


if run_design:
    try:
        early_age_days = AGE_MAP[age_option]

        result = design_mix_from_strengths_min(
            models=models,
            early_min=float(early_strength),
            f28_min=float(f28_min),
            binder_family_key=family,
            early_age_days=int(early_age_days),
            wb_override=None if wb_override is None else float(wb_override),
        )

        pp = result["predicted_parameters"]
        wb = float(pp["water_binder_ratio"])
        binder_total = float(pp["binder_total_kg_m3"])
        water = float(pp["water_kg_m3"])
        ec_total = float(result["embodied_carbon"]["EC_total"])

        pred_f3 = float(predict_f3_from_wb(models, wb, family))
        pred_f7 = float(predict_f7_from_wb(models, wb, family))
        pred_f14 = float(predict_f14_from_wb(models, wb, family))
        pred_f28 = float(implied_f28_from_wb(models, wb, family))

        result_df = format_mix_output(result)

        tab1, tab2, tab3 = st.tabs(["Summary", "Mix Design", "Strengths"])

        with tab1:
            st.subheader("Design Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Concrete family", family)
            c2.metric("Estimated w/b ratio", f"{wb:.3f}")
            c3.metric("Water", f"{water:.1f} kg/m³")
            c4.metric("Total binder", f"{binder_total:.1f} kg/m³")

            c5, c6, c7 = st.columns(3)
            c5.metric("28-day requirement", f"≥ {f28_min:.1f} MPa")
            c6.metric("Early-age requirement", f"≥ {early_strength:.1f} MPa")
            c7.metric("Embodied carbon", f"{ec_total:.1f} kgCO₂-e/m³")

            st.success(
                f"Mix design generated for {family} with a {early_age_days}-day minimum strength requirement."
            )

        with tab2:
            st.subheader("Mix Design Output")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download mix design as CSV",
                data=csv,
                file_name=f"mix_design_{family}_{early_age_days}d.csv",
                mime="text/csv",
            )

        with tab3:
            st.subheader("Predicted Strengths")

            strength_df = pd.DataFrame(
                {
                    "Age": ["3 Day", "7 Day", "14 Day", "28 Day"],
                    "Predicted strength (MPa)": [
                        round(pred_f3, 2),
                        round(pred_f7, 2),
                        round(pred_f14, 2),
                        round(pred_f28, 2),
                    ],
                }
            )
            st.table(strength_df)

            st.markdown("**Selected design requirement**")
            st.write(f"- Required {early_age_days}-day strength: **{early_strength:.2f} MPa**")
            st.write(f"- Required 28-day strength: **{f28_min:.2f} MPa**")

        with st.expander("Technical details (for research use)"):
            st.write("Model workflow:")
            st.write("1. The selected 28-day requirement is used to estimate the water–binder ratio.")
            st.write("2. The selected early-age requirement and age are used in the mix-design model.")
            st.write("3. Water content is estimated from the early-age strength, 28-day strength, and water–binder ratio.")
            st.write("4. Binder split, admixtures, aggregates, and embodied carbon are then calculated.")
            st.write("5. 3-day, 7-day, 14-day, and 28-day strengths are reported from the final water–binder ratio.")

            st.json(
                {
                    "family": family,
                    "selected_requirement_days": int(early_age_days),
                    "input_early_strength_mpa": float(early_strength),
                    "input_28_day_strength_mpa": float(f28_min),
                    "estimated_wb": round(wb, 4),
                    "predicted_f3_mpa": round(pred_f3, 4),
                    "predicted_f7_mpa": round(pred_f7, 4),
                    "predicted_f14_mpa": round(pred_f14, 4),
                    "predicted_f28_mpa": round(pred_f28, 4),
                }
            )

    except Exception as e:
        st.error("The mix design could not be generated.")
        st.exception(e)
else:
    st.info("Enter the design inputs on the left and click **Generate Mix Design**.")