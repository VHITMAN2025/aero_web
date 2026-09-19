import os
import joblib
import pandas as pd
import numpy as np


# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
# LOAD MODELS
# --------------------------------------------------

def load_models():

    rf_cd = joblib.load(CD_MODEL_PATH)
    rf_cl = joblib.load(CL_MODEL_PATH)

    return rf_cd, rf_cl


# --------------------------------------------------
# FEATURE ORDER
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
# PREDICTION
# --------------------------------------------------

def predict_aerodynamics(values):

    rf_cd, rf_cl = load_models()

    input_df = pd.DataFrame(
        [values],
        columns=FEATURE_COLUMNS
    )

    predicted_cd = rf_cd.predict(input_df)[0]

    predicted_cl = rf_cl.predict(input_df)[0]

    return predicted_cd, predicted_cl


# --------------------------------------------------
# MODEL APPLICABILITY
# --------------------------------------------------

def calculate_applicability(values):

    ranges = [
        (800.0, 1200.0),
        (250.0, 315.0),
        (300.0, 500.0),
        (160.0, 240.0),
        (75.0, 246.201938),
        (26.047227, 216.506351),
        (85.557618, 314.431684),
        (6.328849, 70.410078)
    ]

    scores = []

    for value, (minimum, maximum) in zip(
        values,
        ranges
    ):

        midpoint = (
            minimum + maximum
        ) / 2

        half_range = (
            maximum - minimum
        ) / 2

        distance = abs(
            value - midpoint
        )

        score = max(
            0,
            1 - distance / half_range
        )

        scores.append(score)

    return float(
        np.mean(scores) * 100
    )