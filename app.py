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
# INPUTS
# ---------------------------------------------------------

st.sidebar.header("Inputs")

FAMILY_OPTIONS = ["P1","F2","F4","F5","S3","S5","S6","T1","T2"]

family = st.sidebar.selectbox(
    "Binder family",
    FAMILY_OPTIONS
)

early_age_label = st.sidebar.selectbox(
    "Early-age strength age",
    ["3 Day","7 Day","14 Day"]
)

early_age_days = {
    "3 Day":3,
    "7 Day":7,
    "14 Day":14
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
        family=family,
        early_age_days=early_age_days,
        min_early_strength=min_early_strength,
        min_28_strength=min_28_strength,
    )

    wb = mix["wb"]

    pred_f3 = predict_f3_from_wb(models, wb, family)
    pred_f7 = predict_f7_from_wb(models, wb, family)
    pred_f14 = predict_f14_from_wb(models, wb, family)
    pred_f28 = implied_f28_from_wb(models, wb, family)

    st.header("Generated Mix Design")

    st.dataframe(
        pd.DataFrame([mix])
    )

    st.header("Predicted Strengths")

    strength_df = pd.DataFrame(
        {
            "Age":[3,7,14,28],
            "Strength (MPa)":[pred_f3,pred_f7,pred_f14,pred_f28]
        }
    )

    st.dataframe(strength_df)


# ---------------------------------------------------------
# GRAPH FUNCTION
# ---------------------------------------------------------

def build_strength_curve_df(models, families, age_days):

    wb_values = np.linspace(0.34,0.75,120)

    rows = []

    for fam in families:

        for wb in wb_values:

            if age_days == 3:
                strength = predict_f3_from_wb(models,wb,fam)

            elif age_days == 7:
                strength = predict_f7_from_wb(models,wb,fam)

            elif age_days == 14:
                strength = predict_f14_from_wb(models,wb,fam)

            else:
                strength = implied_f28_from_wb(models,wb,fam)

            rows.append(
                {
                    "Water/Binder ratio":wb,
                    "Compressive strength":strength,
                    "Binder family":fam
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# GRAPH SECTION
# ---------------------------------------------------------

st.header("Strength vs Water–Binder Ratio")

graph_age_label = st.selectbox(
    "Graph strength age",
    ["3 Day","7 Day","14 Day","28 Day"],
    index=3
)

graph_age_days = {
    "3 Day":3,
    "7 Day":7,
    "14 Day":14,
    "28 Day":28
}[graph_age_label]


curve_df = build_strength_curve_df(
    models,
    FAMILY_OPTIONS,
    graph_age_days
)


chart = (
    alt.Chart(curve_df)
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
            "Binder family",
            alt.Tooltip("Water/Binder ratio",format=".3f"),
            alt.Tooltip("Compressive strength",format=".2f")
        ],
    )
    .properties(
        height=450
    )
    .interactive()
)

st.altair_chart(chart,use_container_width=True)