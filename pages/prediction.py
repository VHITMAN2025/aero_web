import streamlit as st
import sys
import os

# Allow importing from project root
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from utils.validation import validate_geometry

from utils.prediction import (
    predict_aerodynamics,
    calculate_applicability
)


# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------

st.title("📊 Aerodynamic Prediction")

st.write(
    """
    Enter the vehicle geometry parameters below.
    AeroAI will use the trained Random Forest surrogate
    models to predict the aerodynamic drag and lift
    coefficients.
    """
)

st.divider()


# --------------------------------------------------
# INPUTS
# --------------------------------------------------

st.subheader("Vehicle Geometry")

col1, col2 = st.columns(2)


with col1:

    body_length = st.number_input(
        "Body Length",
        min_value=800.0,
        max_value=1200.0,
        value=1000.0,
        step=1.0
    )

    body_height = st.number_input(
        "Body Height",
        min_value=250.0,
        max_value=315.0,
        value=282.0,
        step=1.0
    )

    body_width = st.number_input(
        "Body Width",
        min_value=300.0,
        max_value=500.0,
        value=400.0,
        step=1.0
    )

    front_arc_diameter = st.number_input(
        "Front Arc Diameter",
        min_value=160.0,
        max_value=240.0,
        value=200.0,
        step=1.0
    )


with col2:

    slant_angle_length = st.number_input(
        "Slant Angle Length",
        min_value=75.0,
        max_value=246.202,
        value=160.0,
        step=1.0
    )

    slant_angle_height = st.number_input(
        "Slant Angle Height",
        min_value=26.047,
        max_value=216.506,
        value=121.0,
        step=1.0
    )

    slant_surface_length = st.number_input(
        "Slant Surface Length",
        min_value=85.558,
        max_value=314.432,
        value=208.0,
        step=1.0
    )

    slant_angle_degrees = st.number_input(
        "Slant Angle Degrees",
        min_value=6.329,
        max_value=70.410,
        value=36.5,
        step=0.1
    )


# --------------------------------------------------
# PREDICTION BUTTON
# --------------------------------------------------

st.divider()

predict_button = st.button(
    "🚀 Predict Aerodynamics",
    type="primary",
    use_container_width=True
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    values = [
        body_length,
        body_height,
        body_width,
        front_arc_diameter,
        slant_angle_length,
        slant_angle_height,
        slant_surface_length,
        slant_angle_degrees
    ]

    errors = validate_geometry(values)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    if errors:

        st.error(
            "Invalid geometry detected."
        )

        for error in errors:
            st.write(f"• {error}")

        st.stop()

    # --------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------

    try:

        predicted_cd, predicted_cl = (
            predict_aerodynamics(values)
        )

        applicability = calculate_applicability(values)

        st.session_state["applicability"] = applicability

        st.session_state["predicted_cd"] = (
            predicted_cd
        )

        st.session_state["predicted_cl"] = (
            predicted_cl
        )

        st.session_state["current_geometry"] = (
            values
        )

        st.success(
            "Aerodynamic prediction completed successfully."
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if (
    "predicted_cd" in st.session_state
    and "predicted_cl" in st.session_state
):

    st.divider()

    st.subheader("Prediction Results")

    cd = st.session_state["predicted_cd"]
    cl = st.session_state["predicted_cl"]

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Drag Coefficient (Cd)",
            f"{cd:.6f}"
        )

    with col2:

        st.metric(
            "Lift Coefficient (Cl)",
            f"{cl:.6f}"
        )


# --------------------------------------------------
# MODEL APPLICABILITY
# --------------------------------------------------

applicability = st.session_state.get(
    "applicability",
    None
)

if applicability is not None:

    st.divider()

    st.subheader(
        "Model Applicability"
    )

    st.progress(
        min(max(int(applicability), 0), 100)
    )

    if applicability >= 70:

        st.success(
            f"Applicability indicator: "
            f"{applicability:.1f}% — "
            "geometry lies well within the training domain."
        )

    elif applicability >= 40:

        st.warning(
            f"Applicability indicator: "
            f"{applicability:.1f}% — "
            "geometry is within the training domain, "
            "but relatively close to its boundaries."
        )

    else:

        st.error(
            f"Applicability indicator: "
            f"{applicability:.1f}% — "
            "prediction should be treated cautiously."
        )


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------

st.divider()

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.info(
        """
        **Cd Model**

        Algorithm: Random Forest

        Test R²: 0.691

        5-Fold CV R²: 0.765
        """
    )

with col2:

    st.info(
        """
        **Cl Model**

        Algorithm: Random Forest

        Test R²: 0.907

        5-Fold CV R²: 0.811
        """
    )


st.caption(
    """
    These values are predictions from the trained
    machine-learning surrogate models. They should
    not be interpreted as direct CFD results.
    """
)