FEATURE_RANGES = {

    "body-length": (800.0, 1200.0),

    "body-height": (250.0, 315.0),

    "body-width": (300.0, 500.0),

    "front-arc-diameter": (160.0, 240.0),

    "slant-angle-length": (75.0, 246.201938),

    "slant-angle-height": (26.047227, 216.506351),

    "slant-surface-length": (85.557618, 314.431684),

    "slant-angle-degrees": (6.328849, 70.410078)
}


def validate_geometry(values):

    errors = []

    feature_names = list(
        FEATURE_RANGES.keys()
    )

    if len(values) != len(feature_names):

        errors.append(
            "Incorrect number of geometry parameters."
        )

        return errors


    for name, value in zip(
        feature_names,
        values
    ):

        minimum, maximum = FEATURE_RANGES[name]

        if value < minimum or value > maximum:

            errors.append(
                f"{name}: {value:.4f} is outside "
                f"the training range "
                f"({minimum:.4f} – {maximum:.4f})."
            )


    return errors