import os
import pandas as pd


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# --------------------------------------------------
# DATA PATHS
# --------------------------------------------------

PARETO_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "pareto_designs.csv"
)

SELECTED_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "selected_designs.csv"
)

UNCERTAINTY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "uncertainty_results.csv"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

def load_pareto_designs():

    return pd.read_csv(
        PARETO_PATH
    )


def load_selected_designs():

    return pd.read_csv(
        SELECTED_PATH
    )


def load_uncertainty():

    return pd.read_csv(
        UNCERTAINTY_PATH
    )