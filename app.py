import streamlit as st

st.set_page_config(
    page_title="AeroAI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

dashboard = st.Page(
    "pages/dashboard.py",
    title="Dashboard",
    icon="🏠"
)

prediction = st.Page(
    "pages/prediction.py",
    title="Aerodynamic Prediction",
    icon="📊"
)

explainable_ai = st.Page(
    "pages/explainable_ai.py",
    title="Explainable AI",
    icon="🧠"
)

optimization = st.Page(
    "pages/optimization.py",
    title="Design Optimization",
    icon="⚙️"
)

comparison = st.Page(
    "pages/comparison.py",
    title="Design Comparison",
    icon="📈"
)

research = st.Page(
    "pages/research.py",
    title="Research Analytics",
    icon="🔬"
)

pg = st.navigation(
    [
        dashboard,
        prediction,
        explainable_ai,
        optimization,
        comparison,
        research
    ]
)

pg.run()