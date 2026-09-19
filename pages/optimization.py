import streamlit as st
import sys
import os
import pandas as pd
import plotly.express as px


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


# ============================================================
# IMPORT UTILITIES
# ============================================================

from utils.optimization import (
    load_pareto_designs,
    load_selected_designs,
    load_uncertainty
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("⚙️ AI Design Optimization")

st.write(
    """
    Explore Pareto-optimal vehicle geometries generated
    using the trained Random Forest aerodynamic surrogate models.
    """
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

try:

    pareto_df = load_pareto_designs()

    selected_df = load_selected_designs()

    uncertainty_df = load_uncertainty()

except Exception as e:

    st.error(
        f"Unable to load optimization data: {e}"
    )

    st.info(
        """
        Please make sure the following files exist:

        data/processed/pareto_designs.csv

        data/processed/selected_designs.csv

        data/processed/uncertainty_results.csv
        """
    )

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_pareto_columns = [
    "predicted_cd",
    "predicted_cl",
    "abs_cl"
]

missing_columns = [
    column
    for column in required_pareto_columns
    if column not in pareto_df.columns
]

if missing_columns:

    st.error(
        "The Pareto dataset is missing these columns: "
        + ", ".join(missing_columns)
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(pareto_df.columns)
    )

    st.stop()


# ============================================================
# DESIGN SELECTION
# ============================================================

st.subheader(
    "🎯 Select Optimization Objective"
)

design_choice = st.radio(
    "Choose a design objective:",
    [
        "Minimum Drag",
        "Balanced",
        "Minimum Lift"
    ],
    horizontal=True
)

if design_choice == "Minimum Drag":

    recommendation = (
        "Recommended when the primary engineering "
        "objective is reducing aerodynamic drag."
    )

elif design_choice == "Balanced":

    recommendation = (
        "Recommended when a compromise between "
        "drag and lift is required."
    )

else:

    recommendation = (
        "Recommended when minimizing absolute lift "
        "is the primary objective."
    )

st.info(
    f"💡 {recommendation}"
)

# ============================================================
# FIND SELECTED DESIGN
# ============================================================

if "Design" not in selected_df.columns:

    st.error(
        "The selected designs file does not contain "
        "a 'Design' column."
    )

    st.stop()


selected = selected_df[
    selected_df["Design"] == design_choice
]


if selected.empty:

    st.error(
        f"{design_choice} was not found in "
        "selected_designs.csv."
    )

    st.stop()


selected = selected.iloc[0]




# ============================================================
# GET PREDICTED VALUES
# ============================================================

# Your selected_designs.csv uses:
# Cd
# Cl
# Abs_Cl

predicted_cd = float(
    selected["Cd"]
)

predicted_cl = float(
    selected["Cl"]
)

absolute_cl = float(
    selected["Abs_Cl"]
)


# ============================================================
# UNCERTAINTY
# ============================================================

cd_std = None
cl_std = None


if "Design" in uncertainty_df.columns:

    uncertainty = uncertainty_df[
        uncertainty_df["Design"] == design_choice
    ]

    if not uncertainty.empty:

        uncertainty = uncertainty.iloc[0]

        if "Cd Std" in uncertainty_df.columns:

            cd_std = float(
                uncertainty["Cd Std"]
            )

        if "Cl Std" in uncertainty_df.columns:

            cl_std = float(
                uncertainty["Cl Std"]
            )


# ============================================================
# DISPLAY MAIN RESULTS
# ============================================================

st.subheader(
    f"🏎️ {design_choice} Design"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Predicted Cd",
        f"{predicted_cd:.6f}"
    )


with col2:

    st.metric(
        "Predicted Cl",
        f"{predicted_cl:.6f}"
    )


with col3:

    st.metric(
        "Absolute Cl",
        f"{absolute_cl:.6f}"
    )


# ============================================================
# MODEL UNCERTAINTY
# ============================================================

st.divider()

st.subheader(
    "📐 Model Uncertainty"
)


col1, col2 = st.columns(2)


with col1:

    if cd_std is not None:

        st.metric(
            "Cd Standard Deviation",
            f"{cd_std:.6f}"
        )

    else:

        st.metric(
            "Cd Standard Deviation",
            "N/A"
        )


with col2:

    if cl_std is not None:

        st.metric(
            "Cl Standard Deviation",
            f"{cl_std:.6f}"
        )

    else:

        st.metric(
            "Cl Standard Deviation",
            "N/A"
        )


# ============================================================
# ENGINEERING INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "💡 Engineering Interpretation"
)


if design_choice == "Minimum Drag":

    st.success(
        """
        **Minimum Drag Candidate**

        This design prioritizes reduction of aerodynamic
        drag and provides the lowest predicted Cd among
        the selected optimization candidates.
        """
    )


elif design_choice == "Balanced":

    st.warning(
        """
        **Balanced Candidate**

        This design provides a compromise between
        aerodynamic drag and lift and is suitable when
        both objectives need to be considered.
        """
    )


else:

    st.info(
        """
        **Minimum Lift Candidate**

        This design prioritizes minimizing the absolute
        lift coefficient, while accepting a higher
        predicted drag coefficient.
        """
    )


# ============================================================
# OPTIMIZED VEHICLE GEOMETRY
# ============================================================

st.divider()

st.subheader(
    "📏 Optimized Vehicle Geometry"
)


geometry_columns = [
    "body-length",
    "body-height",
    "body-width",
    "front-arc-diameter",
    "slant-angle-length",
    "slant-angle-height",
    "slant-surface-length",
    "slant-angle-degrees"
]


# ------------------------------------------------------------
# CHECK IF SELECTED FILE ALREADY CONTAINS GEOMETRY
# ------------------------------------------------------------

has_geometry = all(
    column in selected.index
    for column in geometry_columns
)


if has_geometry:

    geometry = selected

else:

    # --------------------------------------------------------
    # MATCH SELECTED DESIGN TO PARETO DATA
    # --------------------------------------------------------

    geometry = None

    if (
        "predicted_cd" in pareto_df.columns
        and "predicted_cl" in pareto_df.columns
    ):

        cd_difference = (
            pareto_df["predicted_cd"]
            - predicted_cd
        ).abs()

        cl_difference = (
            pareto_df["predicted_cl"]
            - predicted_cl
        ).abs()

        total_difference = (
            cd_difference
            + cl_difference
        )

        closest_index = (
            total_difference
            .idxmin()
        )

        geometry = pareto_df.loc[
            closest_index
        ]


# ------------------------------------------------------------
# DISPLAY GEOMETRY
# ------------------------------------------------------------

if geometry is not None:

    geometry_data = []

    for column in geometry_columns:

        if column in geometry.index:

            geometry_data.append({

                "Parameter": column,

                "Value": float(
                    geometry[column]
                )

            })


    geometry_display = pd.DataFrame(
        geometry_data
    )


    geometry_display["Value"] = (
        geometry_display["Value"]
        .round(4)
    )


    st.dataframe(
        geometry_display,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Optimized geometry could not be matched "
        "to the Pareto dataset."
    )


# ============================================================
# PARETO-OPTIMAL DESIGN SPACE
# ============================================================

st.divider()

st.subheader(
    "📊 Pareto-Optimal Design Space"
)

st.write(
    """
    Each point represents a candidate vehicle geometry.
    The horizontal axis represents predicted drag, while
    the vertical axis represents absolute lift.
    Lower values are generally preferred for both objectives.
    """
)


# ------------------------------------------------------------
# CREATE PLOT DATA
# ------------------------------------------------------------

plot_df = pareto_df.copy()


# Add design labels

plot_df["Design Type"] = "Pareto Candidate"


# ------------------------------------------------------------
# CREATE INTERACTIVE PLOTLY CHART
# ------------------------------------------------------------

hover_columns = [
    column
    for column in [
        "predicted_cl",
        "body-length",
        "body-height",
        "body-width",
        "front-arc-diameter",
        "slant-angle-length",
        "slant-angle-height",
        "slant-surface-length",
        "slant-angle-degrees"
    ]
    if column in plot_df.columns
]


fig = px.scatter(

    plot_df,

    x="predicted_cd",

    y="abs_cl",

    hover_data=hover_columns,

    labels={

        "predicted_cd":
            "Predicted Drag Coefficient (Cd)",

        "abs_cl":
            "Absolute Lift Coefficient (|Cl|)",

        "predicted_cl":
            "Predicted Lift Coefficient (Cl)"

    },

    title="AI Pareto Design Space"
)


# ============================================================
# HIGHLIGHT SELECTED DESIGN
# ============================================================

selected_x = predicted_cd

selected_y = absolute_cl


fig.add_scatter(

    x=[selected_x],

    y=[selected_y],

    mode="markers",

    marker=dict(
        size=16,
        symbol="star"
    ),

    name=design_choice,

    hovertemplate=(
        f"<b>{design_choice}</b><br>"
        f"Cd = {selected_x:.6f}<br>"
        f"|Cl| = {selected_y:.6f}"
        "<extra></extra>"
    )
)


# ============================================================
# CHART SETTINGS
# ============================================================

fig.update_layout(

    height=600,

    xaxis_title=(
        "Predicted Drag Coefficient (Cd)"
    ),

    yaxis_title=(
        "Absolute Lift Coefficient (|Cl|)"
    ),

    hovermode="closest"
)


st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# PARETO DATA TABLE
# ============================================================

st.divider()

st.subheader(
    "📋 All Pareto-Optimal Designs"
)


display_columns = [

    "predicted_cd",

    "predicted_cl",

    "abs_cl"

]


available_columns = [

    column

    for column in display_columns

    if column in pareto_df.columns

]


pareto_display = pareto_df[
    available_columns
].copy()


for column in available_columns:

    pareto_display[column] = (
        pareto_display[column]
        .astype(float)
        .round(6)
    )


st.dataframe(

    pareto_display,

    use_container_width=True,

    hide_index=True
)


# ============================================================
# AI RECOMMENDATION
# ============================================================

st.divider()

st.subheader(
    "⭐ AI Engineering Recommendation"
)


if design_choice == "Minimum Drag":

    recommendation_text = (
        "The Minimum Drag design is recommended "
        "when reducing aerodynamic drag is the "
        "primary engineering objective."
    )


elif design_choice == "Balanced":

    recommendation_text = (
        "The Balanced design is recommended when "
        "both drag and lift need to be considered "
        "simultaneously."
    )


else:

    recommendation_text = (
        "The Minimum Lift design is recommended "
        "when minimizing absolute lift is the "
        "primary engineering objective."
    )


st.success(
    recommendation_text
)


# ============================================================
# IMPORTANT ENGINEERING DISCLAIMER
# ============================================================

st.divider()

st.warning(
    """
    **Engineering Validation Notice**

    These optimized aerodynamic coefficients are generated
    by machine-learning surrogate models trained using
    AhmedML data.

    The optimized geometries should be treated as candidate
    designs for early-stage engineering exploration.

    CFD validation or experimental testing is recommended
    before using a newly generated geometry for real-world
    engineering deployment.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    """
    AeroAI — AI-Based Vehicle Aerodynamic Performance
    Prediction and Design Optimization
    """
)