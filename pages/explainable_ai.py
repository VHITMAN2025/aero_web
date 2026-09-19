import streamlit as st
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)


from utils.shap_analysis import (
    calculate_shap_values,
    FEATURE_COLUMNS
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("🧠 Explainable AI")

st.write(
    """
    Understand why the Random Forest model produces
    a particular aerodynamic prediction.
    """
)

st.divider()


# --------------------------------------------------
# CHECK FOR CURRENT GEOMETRY
# --------------------------------------------------

if "current_geometry" not in st.session_state:

    st.warning(
        """
        No vehicle geometry has been analyzed yet.

        Go to **Aerodynamic Prediction**, enter the
        vehicle geometry, and click **Predict Aerodynamics**.
        """
    )

    st.stop()


# --------------------------------------------------
# CURRENT GEOMETRY
# --------------------------------------------------

geometry = st.session_state[
    "current_geometry"
]

predicted_cd = st.session_state.get(
    "predicted_cd",
    None
)

predicted_cl = st.session_state.get(
    "predicted_cl",
    None
)


# --------------------------------------------------
# CURRENT PREDICTION
# --------------------------------------------------

st.subheader(
    "Current Aerodynamic Prediction"
)

col1, col2 = st.columns(2)

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


st.divider()


# --------------------------------------------------
# TARGET SELECTION
# --------------------------------------------------

target = st.radio(
    "Select aerodynamic coefficient to explain:",
    ["Cd", "Cl"],
    horizontal=True
)


# --------------------------------------------------
# SHAP CALCULATION
# --------------------------------------------------

try:

    shap_values = calculate_shap_values(
        geometry,
        target
    )

except Exception as e:

    st.error(
        f"SHAP calculation failed: {e}"
    )

    st.stop()


# --------------------------------------------------
# SHAP DATAFRAME
# --------------------------------------------------

shap_df = pd.DataFrame({

    "Feature": FEATURE_COLUMNS,

    "Value": geometry,

    "SHAP Value": shap_values

})

shap_df["Absolute Impact"] = (
    shap_df["SHAP Value"].abs()
)

shap_df = shap_df.sort_values(
    "Absolute Impact",
    ascending=False
)


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

st.subheader(
    f"Feature Contributions to {target}"
)

st.write(
    """
    A positive SHAP value pushes the prediction
    higher, while a negative SHAP value pushes the
    prediction lower relative to the model's baseline.
    """
)


# --------------------------------------------------
# BAR CHART
# --------------------------------------------------

plot_df = shap_df.sort_values(
    "SHAP Value"
)

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.barh(
    plot_df["Feature"],
    plot_df["SHAP Value"]
)

ax.axvline(
    0,
    linewidth=1
)

ax.set_xlabel(
    "SHAP Contribution"
)

ax.set_ylabel(
    "Vehicle Parameter"
)

ax.set_title(
    f"SHAP Explanation — {target}"
)

plt.tight_layout()

st.pyplot(fig)


# --------------------------------------------------
# CONTRIBUTION TABLE
# --------------------------------------------------

st.subheader(
    "Feature Contribution Details"
)

display_df = shap_df.copy()

display_df["Value"] = display_df[
    "Value"
].round(4)

display_df["SHAP Value"] = display_df[
    "SHAP Value"
].round(6)

display_df["Absolute Impact"] = display_df[
    "Absolute Impact"
].round(6)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# TOP FEATURES
# --------------------------------------------------

st.subheader(
    f"Most Influential Parameters for {target}"
)

top_features = shap_df.head(3)

for _, row in top_features.iterrows():

    direction = (
        "increases"
        if row["SHAP Value"] > 0
        else "decreases"
    )

    st.write(
        f"**{row['Feature']}** "
        f"({row['Value']:.3f}) → "
        f"{direction} the predicted {target} "
        f"by approximately "
        f"{abs(row['SHAP Value']):.6f} "
        f"relative to the model baseline."
    )


# --------------------------------------------------
# ENGINEERING NOTE
# --------------------------------------------------

st.divider()

st.info(
    """
    **Engineering interpretation**

    SHAP explains the contribution of each input
    parameter to the machine-learning prediction.
    It does not prove a physical cause-and-effect
    relationship. The explanation describes the
    behavior of the trained surrogate model.
    """
)