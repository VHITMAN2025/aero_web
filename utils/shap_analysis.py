import os
import joblib
import pandas as pd
import numpy as np
import shap


# --------------------------------------------------
# PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

CD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rf_cd.pkl"
)

CL_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "rf_cl.pkl"
)


# --------------------------------------------------
# FEATURE NAMES
# --------------------------------------------------

FEATURE_COLUMNS = [
    "body-length",
    "body-height",
    "body-width",
    "front-arc-diameter",
    "slant-angle-length",
    "slant-angle-height",
    "slant-surface-length",
    "slant-angle-degrees"
]


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------

def load_models():

    rf_cd = joblib.load(
        CD_MODEL_PATH
    )

    rf_cl = joblib.load(
        CL_MODEL_PATH
    )

    return rf_cd, rf_cl


# --------------------------------------------------
# CALCULATE SHAP VALUES
# --------------------------------------------------

def calculate_shap_values(
    input_values,
    target="Cd"
):

    rf_cd, rf_cl = load_models()

    if target == "Cd":
        model = rf_cd

    elif target == "Cl":
        model = rf_cl

    else:
        raise ValueError(
            "Target must be either 'Cd' or 'Cl'."
        )

    input_df = pd.DataFrame(
        [input_values],
        columns=FEATURE_COLUMNS
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        input_df
    )

    # For regression, SHAP returns:
    # [number_of_samples, number_of_features]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 1:
        values = shap_values

    else:
        values = shap_values[0]

    return values