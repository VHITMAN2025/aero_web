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
# PAGE TITLE
# --------------------------------------------------

st.title("🔬 Research Analytics")

st.write(
    """
    Research-oriented summary of the AeroAI aerodynamic
    prediction and design optimization framework.
    """
)

st.divider()


# ==================================================
# 1. DATASET OVERVIEW
# ==================================================

st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Training Samples",
        "499"
    )

with col2:
    st.metric(
        "Input Features",
        "8"
    )

with col3:
    st.metric(
        "Prediction Targets",
        "2"
    )

with col4:
    st.metric(
        "Optimization Candidates",
        "49,756"
    )


st.info(
    """
    The aerodynamic dataset contains eight vehicle geometry
    parameters used as input features and two aerodynamic
    response variables: Drag Coefficient (Cd) and Lift
    Coefficient (Cl).
    """
)


# ==================================================
# 2. INPUT FEATURES
# ==================================================

st.subheader("Input Geometry Parameters")

feature_df = pd.DataFrame({
    "Feature": [
        "body-length",
        "body-height",
        "body-width",
        "front-arc-diameter",
        "slant-angle-length",
        "slant-angle-height",
        "slant-surface-length",
        "slant-angle-degrees"
    ],

    "Role": [
        "Vehicle body length",
        "Vehicle body height",
        "Vehicle body width",
        "Front curvature diameter",
        "Slant region length",
        "Slant region height",
        "Slant surface length",
        "Slant angle"
    ]
})

st.dataframe(
    feature_df,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 3. MODEL COMPARISON
# ==================================================

st.divider()

st.header("2. Machine Learning Model Performance")

model_results = pd.DataFrame({

    "Model": [
        "Linear Regression",
        "Linear Regression",
        "Random Forest",
        "Random Forest",
        "XGBoost",
        "XGBoost"
    ],

    "Target": [
        "Cd",
        "Cl",
        "Cd",
        "Cl",
        "Cd",
        "Cl"
    ],

    "MAE": [
        0.028865,
        0.136394,
        0.020764,
        0.044822,
        0.021640,
        0.045446
    ],

    "RMSE": [
        0.039727,
        0.165683,
        0.033756,
        0.072183,
        0.034021,
        0.072802
    ],

    "R²": [
        0.572640,
        0.509036,
        0.691453,
        0.906813,
        0.686576,
        0.905206
    ]
})

st.dataframe(
    model_results,
    use_container_width=True,
    hide_index=True
)


# --------------------------------------------------
# BEST MODEL
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.success(
        """
        ### Best Cd Model

        **Random Forest**

        R² = **0.691453**
        """
    )

with col2:

    st.success(
        """
        ### Best Cl Model

        **Random Forest**

        R² = **0.906813**
        """
    )


# ==================================================
# 4. CROSS VALIDATION
# ==================================================

st.divider()

st.header("3. Five-Fold Cross Validation")

cv_results = pd.DataFrame({

    "Target": [
        "Cd",
        "Cl"
    ],

    "MAE Mean": [
        0.018361,
        0.051290
    ],

    "MAE Std": [
        0.001768,
        0.003906
    ],

    "RMSE Mean": [
        0.028980,
        0.092332
    ],

    "RMSE Std": [
        0.002135,
        0.006611
    ],

    "R² Mean": [
        0.765182,
        0.810764
    ],

    "R² Std": [
        0.034974,
        0.026849
    ]
})

st.dataframe(
    cv_results,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 5. HYPERPARAMETER TUNING
# ==================================================

st.divider()

st.header("4. Hyperparameter Optimization")

st.write(
    """
    Random Forest hyperparameters were optimized using
    cross-validation with RMSE as the optimization metric.
    """
)

hyperparameter_df = pd.DataFrame({

    "Parameter": [
        "n_estimators",
        "max_depth",
        "max_features",
        "min_samples_leaf",
        "min_samples_split"
    ],

    "Best Value": [
        "227",
        "None",
        "1.0",
        "2",
        "5"
    ]
})

st.dataframe(
    hyperparameter_df,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 6. SHAP / EXPLAINABILITY
# ==================================================

st.divider()

st.header("5. Explainable AI")

st.write(
    """
    SHAP-based analysis is used to explain the contribution
    of geometry parameters to the surrogate-model predictions.
    """
)

st.info(
    """
    The Explainable AI module provides both global model
    interpretation and local explanations for individual
    vehicle configurations.
    """
)

st.markdown(
    """
    **Available explanations**

    - Global feature importance
    - Local Cd prediction explanation
    - Local Cl prediction explanation
    - Positive and negative feature contributions
    """
)


# ==================================================
# 7. OPTIMIZATION
# ==================================================

st.divider()

st.header("6. Design Optimization")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Valid Candidate Designs",
        "49,756"
    )

with col2:

    st.metric(
        "Pareto-Optimal Designs",
        "11"
    )

with col3:

    st.metric(
        "Optimization Objectives",
        "2"
    )


st.write(
    """
    Candidate geometries were evaluated using the trained
    Random Forest surrogate models. Physically valid designs
    were retained and Pareto optimization was applied to
    identify trade-offs between drag and lift.
    """
)


# ==================================================
# 8. FINAL CANDIDATES
# ==================================================

st.divider()

st.header("7. Final AI-Generated Candidates")

try:

    selected_df = load_selected_designs()
    uncertainty_df = load_uncertainty()

except Exception as e:

    st.error(
        f"Unable to load candidate results: {e}"
    )

    st.stop()


final_df = selected_df.copy()

final_df = final_df.rename(
    columns={
        "Cd": "Predicted Cd",
        "Cl": "Predicted Cl",
        "Abs_Cl": "Absolute Cl"
    }
)


# Add uncertainty

final_df["Cd Std"] = np.nan
final_df["Cl Std"] = np.nan

for _, row in uncertainty_df.iterrows():

    mask = (
        final_df["Design"]
        == row["Design"]
    )

    final_df.loc[
        mask,
        "Cd Std"
    ] = row["Cd Std"]

    final_df.loc[
        mask,
        "Cl Std"
    ] = row["Cl Std"]


for column in [
    "Predicted Cd",
    "Predicted Cl",
    "Absolute Cl",
    "Cd Std",
    "Cl Std"
]:

    if column in final_df.columns:

        final_df[column] = (
            final_df[column]
            .astype(float)
            .round(6)
        )


st.dataframe(
    final_df,
    use_container_width=True,
    hide_index=True
)


# ==================================================
# 9. RECOMMENDATION
# ==================================================

st.divider()

st.header("8. AI Engineering Recommendation")

st.success(
    """
    ### ⭐ Primary Candidate — Minimum Drag

    **Predicted Cd:** 0.192379

    **Predicted Cl:** 0.023576

    **Cd uncertainty:** ±0.011521

    The Minimum Drag candidate is selected as the
    primary AI recommendation because it provides the
    lowest predicted drag among the selected designs
    and the lowest Cd uncertainty.
    """
)


st.warning(
    """
    ### ⚠️ Validation Requirement

    The reported aerodynamic coefficients for newly
    generated geometries are surrogate-model predictions.
    CFD validation is recommended before engineering
    deployment.
    """
)


# ==================================================
# 10. RESEARCH LIMITATIONS
# ==================================================

st.divider()

st.header("9. Research Limitations")

st.markdown(
    """
    **Current limitations**

    1. The machine-learning models are trained on
       AhmedML aerodynamic data.

    2. Optimized geometries are surrogate-model
       predictions and have not been directly simulated
       using new CFD runs.

    3. Random Forest ensemble variation is used as an
       uncertainty indicator, rather than a calibrated
       probabilistic confidence interval.

    4. The optimized designs should therefore be treated
       as candidate geometries for further engineering
       validation.
    """
)


# ==================================================
# 11. FUTURE WORK
# ==================================================

st.divider()

st.header("10. Future Work")

st.markdown(
    """
    **Potential extensions**

    - Direct CFD validation of optimized geometries
    - More advanced aerodynamic datasets
    - Deep-learning surrogate models
    - Physics-informed machine learning
    - 3D geometry generation
    - Automated CAD integration
    - Multi-fidelity CFD/ML modeling
    - Real-time engineering optimization
    """
)


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    """
    AeroAI — AI-Based Vehicle Aerodynamic Performance
    Prediction and Design Optimization
    """
)