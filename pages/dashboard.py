import streamlit as st

st.title("✈️ AeroAI")

st.subheader(
    "Intelligent Vehicle Aerodynamic "
    "Design & Optimization Platform"
)

st.write(
    """
    Welcome to AeroAI.

    This platform uses machine learning,
    explainable AI, and multi-objective optimization
    to support early-stage vehicle aerodynamic design.
    """
)

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Training Samples",
        "499"
    )

with col2:
    st.metric(
        "Geometry Features",
        "8"
    )

with col3:
    st.metric(
        "Valid Candidates",
        "49,756"
    )

with col4:
    st.metric(
        "Pareto Designs",
        "11"
    )

st.divider()

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Random Forest — Cd R²",
        "0.691"
    )

with col2:

    st.metric(
        "Random Forest — Cl R²",
        "0.907"
    )

st.divider()

st.subheader("Primary AI Candidate")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicted Cd",
        "0.192379"
    )

with col2:

    st.metric(
        "Predicted Cl",
        "0.023576"
    )

st.info(
    """
    **Primary candidate:** Minimum Drag Design.

    The values shown are Random Forest surrogate-model
    predictions and should be CFD-validated before engineering use.
    """
)