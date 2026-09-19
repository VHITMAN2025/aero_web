import streamlit as st
import pandas as pd
import numpy as np
import os
import sys


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from utils.optimization import (
    load_pareto_designs,
    load_selected_designs,
    load_uncertainty
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.title("📈 Design Comparison")

st.write(
    """
    Compare the baseline AhmedML design with the
    AI-generated Pareto-optimal designs.
    """
)

st.divider()


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

try:

    pareto_df = load_pareto_designs()
    selected_df = load_selected_designs()
    uncertainty_df = load_uncertainty()

except Exception as e:

    st.error(
        f"Unable to load comparison data: {e}"
    )

    st.stop()


# --------------------------------------------------
# BASELINE DATA
# --------------------------------------------------

baseline = {
    "Design": "Baseline",
    "Cd": 0.19318417212586733,
    "Cl": 0.036594816539258224,
    "Abs_Cl": abs(0.036594816539258224)
}


# --------------------------------------------------
# SELECTED DESIGNS
# --------------------------------------------------

comparison_rows = [
    baseline
]

for _, row in selected_df.iterrows():

    comparison_rows.append({
        "Design": row["Design"],
        "Cd": row["Cd"],
        "Cl": row["Cl"],
        "Abs_Cl": row["Abs_Cl"]
    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# --------------------------------------------------
# UNCERTAINTY
# --------------------------------------------------

comparison_df["Cd Std"] = np.nan
comparison_df["Cl Std"] = np.nan

for _, row in uncertainty_df.iterrows():

    mask = (
        comparison_df["Design"]
        == row["Design"]
    )

    comparison_df.loc[
        mask,
        "Cd Std"
    ] = row["Cd Std"]

    comparison_df.loc[
        mask,
        "Cl Std"
    ] = row["Cl Std"]


# --------------------------------------------------
# MAIN RESULTS TABLE
# --------------------------------------------------

st.subheader(
    "Aerodynamic Performance"
)

display_df = comparison_df.copy()

display_df["Cd"] = display_df["Cd"].round(6)
display_df["Cl"] = display_df["Cl"].round(6)
display_df["Abs_Cl"] = display_df["Abs_Cl"].round(6)
display_df["Cd Std"] = display_df["Cd Std"].round(6)
display_df["Cl Std"] = display_df["Cl Std"].round(6)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# IMPROVEMENT CALCULATIONS
# --------------------------------------------------

baseline_cd = baseline["Cd"]
baseline_abs_cl = baseline["Abs_Cl"]

improvements = []

for _, row in comparison_df.iterrows():

    if row["Design"] == "Baseline":
        continue

    cd_change = (
        (row["Cd"] - baseline_cd)
        / baseline_cd
    ) * 100

    cl_improvement = (
        (baseline_abs_cl - row["Abs_Cl"])
        / baseline_abs_cl
    ) * 100

    improvements.append({
        "Design": row["Design"],
        "Cd Change (%)": cd_change,
        "|Cl| Improvement (%)": cl_improvement
    })


improvement_df = pd.DataFrame(
    improvements
)


# --------------------------------------------------
# IMPROVEMENT TABLE
# --------------------------------------------------

st.divider()

st.subheader(
    "Change Relative to Baseline"
)

improvement_display = (
    improvement_df.copy()
)

improvement_display[
    "Cd Change (%)"
] = improvement_display[
    "Cd Change (%)"
].round(2)

improvement_display[
    "|Cl| Improvement (%)"
] = improvement_display[
    "|Cl| Improvement (%)"
].round(2)

st.dataframe(
    improvement_display,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# INTERPRETATION
# --------------------------------------------------

st.divider()

st.subheader(
    "Engineering Interpretation"
)

st.markdown(
    """
    **Minimum Drag**

    Provides the lowest predicted drag coefficient
    among the selected AI designs.

    **Balanced**

    Provides a compromise between drag and lift,
    with a substantially lower predicted absolute
    lift coefficient than the baseline.

    **Minimum Lift**

    Produces the lowest predicted absolute lift,
    but with a higher predicted drag coefficient.
    """
)


# --------------------------------------------------
# PERFORMANCE CHARTS
# --------------------------------------------------

st.divider()

st.subheader(
    "Performance Visualization"
)

chart_df = comparison_df[
    ["Design", "Cd", "Abs_Cl"]
].copy()

chart_df = chart_df.set_index(
    "Design"
)


# --------------------------------------------------
# CD CHART
# --------------------------------------------------

st.write("### Drag Coefficient — Cd")

st.bar_chart(
    chart_df["Cd"]
)


# --------------------------------------------------
# CL CHART
# --------------------------------------------------

st.write("### Absolute Lift Coefficient — |Cl|")

st.bar_chart(
    chart_df["Abs_Cl"]
)


# --------------------------------------------------
# UNCERTAINTY
# --------------------------------------------------

st.divider()

st.subheader(
    "Model Uncertainty"
)

uncertainty_chart = comparison_df[
    [
        "Design",
        "Cd Std",
        "Cl Std"
    ]
].copy()

uncertainty_chart = uncertainty_chart.set_index(
    "Design"
)

st.bar_chart(
    uncertainty_chart
)


# --------------------------------------------------
# AUTOMATIC RECOMMENDATION
# --------------------------------------------------

st.divider()

st.subheader(
    "⭐ AI Engineering Recommendation"
)


minimum_drag = comparison_df[
    comparison_df["Design"] == "Minimum Drag"
].iloc[0]

balanced = comparison_df[
    comparison_df["Design"] == "Balanced"
].iloc[0]

minimum_lift = comparison_df[
    comparison_df["Design"] == "Minimum Lift"
].iloc[0]


st.success(
    f"""
    **Primary Candidate: Minimum Drag Design**

    Predicted Cd: **{minimum_drag["Cd"]:.6f}**

    Predicted Cl: **{minimum_drag["Cl"]:.6f}**

    Cd uncertainty: **±{minimum_drag["Cd Std"]:.6f}**

    This candidate provides the lowest predicted drag
    among the selected optimized designs and has the
    lowest Cd uncertainty of the three candidates.
    """
)


st.warning(
    """
    **Important:** These values are surrogate-model
    predictions. Direct CFD validation is recommended
    before engineering deployment of a newly generated
    geometry.
    """
)


# --------------------------------------------------
# REPORT SUMMARY
# --------------------------------------------------

st.divider()

st.subheader(
    "Final Comparison Summary"
)

st.write(
    """
    The AI optimization framework identifies different
    candidate designs according to engineering objectives.
    The Minimum Drag design prioritizes drag reduction,
    the Balanced design prioritizes a Cd–Cl trade-off,
    and the Minimum Lift design prioritizes minimizing
    absolute lift.
    """
)