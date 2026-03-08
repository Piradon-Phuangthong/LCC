import streamlit as st
import pandas as pd

from lcc.dataset import build_df
from lcc.models import (
    build_models,
    predict_f3_from_wb,
    predict_f7_from_wb,
    implied_f28_from_wb,
    predict_wb_from_f28_curve,
)
from lcc.design import design_mix_from_strengths_min


# ------------------------------------------------------------
# PAGE SETUP
# ------------------------------------------------------------
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
2. Choose whether your requirement is based on **3-day** or **7-day** strength
3. Enter the **required compressive strength**
4. Click **Generate Mix Design**

The tool will estimate:
- water–binder ratio
- 3-day, 7-day, and 28-day strengths
- mix quantities
- embodied carbon (if available in the result output)
"""
)

st.divider()


# ------------------------------------------------------------
# LOAD DATA / MODELS
# ------------------------------------------------------------
@st.cache_resource
def load_models():
    df = build_df()
    models = build_models(df)
    return df, models


df, models = load_models()


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
FAMILY_OPTIONS = ["P1", "F2", "F4", "F5", "S3", "S5", "S6", "T1", "T2"]

NUMERIC_PRIORITY_KEYS = [
    "Water",
    "Binder",
    "Total Binder",
    "Cement",
    "GGBFS",
    "Fly Ash",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Sand",
    "Stone",
    "Superplasticizer",
    "Admixture",
    "EC_Total",
]


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def find_first_key(result_dict, possible_keys):
    lower_map = {str(k).strip().lower(): k for k in result_dict.keys()}
    for key in possible_keys:
        actual = lower_map.get(key.lower())
        if actual is not None:
            return actual
    return None


def format_result_dataframe(result_dict):
    rows = []
    for k, v in result_dict.items():
        num = safe_float(v)
        if num is not None:
            rows.append({"Material / Output": k, "Value": round(num, 3)})
        else:
            rows.append({"Material / Output": k, "Value": v})

    result_df = pd.DataFrame(rows)

    def sort_key(row_name):
        if row_name in NUMERIC_PRIORITY_KEYS:
            return NUMERIC_PRIORITY_KEYS.index(row_name)
        return len(NUMERIC_PRIORITY_KEYS) + 100

    result_df["_sort"] = result_df["Material / Output"].apply(sort_key)
    result_df = result_df.sort_values(by=["_sort", "Material / Output"]).drop(columns="_sort")
    return result_df


# ------------------------------------------------------------
# SIDEBAR INPUTS
# ------------------------------------------------------------
st.sidebar.header("Design Inputs")

family = st.sidebar.selectbox(
    "Concrete family",
    FAMILY_OPTIONS,
    index=0,
)

age_option = st.sidebar.selectbox(
    "Required early-age strength",
    ["3 Day", "7 Day"],
    index=1,
)

strength = st.sidebar.number_input(
    "Required compressive strength (MPa)",
    min_value=1.0,
    max_value=100.0,
    value=30.0,
    step=1.0,
)

run_design = st.sidebar.button("Generate Mix Design", type="primary")


# ------------------------------------------------------------
# MAIN ACTION
# ------------------------------------------------------------
if run_design:
    try:
        f3_target = None
        f7_target = None

        if age_option == "3 Day":
            f3_target = float(strength)
        else:
            f7_target = float(strength)

        # Estimate 28-day strength from selected early-age target
        if f3_target is not None:
            f28 = implied_f28_from_wb(models, f3=f3_target)
        else:
            f28 = implied_f28_from_wb(models, f7=f7_target)

        # Estimate water-binder ratio from 28-day strength
        wb = predict_wb_from_f28_curve(models, f28)

        # Predict 3-day and 7-day strengths from w/b
        pred_f3 = predict_f3_from_wb(models, wb)
        pred_f7 = predict_f7_from_wb(models, wb)

        # Generate final mix design
        result = design_mix_from_strengths_min(
            family=family,
            f3=pred_f3,
            f7=pred_f7,
            f28=f28,
            wb=wb,
        )

        # Pull embodied carbon if present
        ec_key = find_first_key(result, ["EC_Total", "ec_total", "Embodied Carbon", "EC total"])
        ec_total = safe_float(result.get(ec_key)) if ec_key else None

        # Pull binder total if present
        binder_key = find_first_key(result, ["Binder", "Total Binder", "binder", "total binder"])
        binder_total = safe_float(result.get(binder_key)) if binder_key else None

        result_df = format_result_dataframe(result)

        # ----------------------------------------------------
        # OUTPUT TABS
        # ----------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["Summary", "Mix Design", "Strengths"])

        with tab1:
            st.subheader("Design Summary")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Concrete family", family)
            c2.metric("Estimated w/b ratio", f"{wb:.3f}")
            c3.metric("Estimated 28-day strength", f"{f28:.2f} MPa")
            c4.metric(
                "Estimated embodied carbon",
                f"{ec_total:.1f} kgCO₂-e/m³" if ec_total is not None else "N/A"
            )

            c5, c6 = st.columns(2)
            c5.metric("Predicted 3-day strength", f"{pred_f3:.2f} MPa")
            c6.metric("Predicted 7-day strength", f"{pred_f7:.2f} MPa")

            if binder_total is not None:
                st.info(f"Estimated total binder: **{binder_total:.1f} kg/m³**")

        with tab2:
            st.subheader("Mix Design Output")
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download mix design as CSV",
                data=csv,
                file_name=f"mix_design_{family}_{age_option.replace(' ', '_').lower()}.csv",
                mime="text/csv",
            )

        with tab3:
            st.subheader("Predicted Strengths")

            strength_df = pd.DataFrame(
                {
                    "Age": ["3 Day", "7 Day", "28 Day"],
                    "Strength (MPa)": [
                        round(float(pred_f3), 2),
                        round(float(pred_f7), 2),
                        round(float(f28), 2),
                    ],
                }
            )
            st.table(strength_df)

            st.markdown("**Selected design requirement**")
            if age_option == "3 Day":
                st.write(f"- Required 3-day strength: **{strength:.2f} MPa**")
            else:
                st.write(f"- Required 7-day strength: **{strength:.2f} MPa**")

        with st.expander("Technical details (for research use)"):
            st.write("Model workflow:")
            st.write("1. Required early-age strength is used to infer an equivalent 28-day strength.")
            st.write("2. 28-day strength is used to estimate the water–binder ratio.")
            st.write("3. The water–binder ratio is used to predict 3-day and 7-day strengths.")
            st.write("4. The selected concrete family and predicted strengths are passed to the mix-design function.")
            st.write("5. The final mix quantities and embodied carbon are displayed if available.")

            st.write("Intermediate values:")
            st.json(
                {
                    "family": family,
                    "selected_requirement": age_option,
                    "input_strength_mpa": float(strength),
                    "estimated_wb": round(float(wb), 4),
                    "predicted_f3_mpa": round(float(pred_f3), 4),
                    "predicted_f7_mpa": round(float(pred_f7), 4),
                    "estimated_f28_mpa": round(float(f28), 4),
                }
            )

    except Exception as e:
        st.error("The mix design could not be generated.")
        st.exception(e)

else:
    st.info("Enter the design inputs on the left and click **Generate Mix Design**.")