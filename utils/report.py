from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER


def create_prediction_report(
    output_path,
    geometry,
    predicted_cd,
    predicted_cl
):

    styles = getSampleStyleSheet()

    document = SimpleDocTemplate(
        output_path,
        pagesize=A4
    )

    elements = []

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    elements.append(
        Paragraph(
            "AeroAI — Aerodynamic Analysis Report",
            title_style
        )
    )

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "AI-Based Vehicle Aerodynamic "
            "Performance Prediction and Design Optimization",
            styles["Heading2"]
        )
    )

    elements.append(
        Spacer(1, 15)
    )

    elements.append(
        Paragraph(
            "Predicted Aerodynamic Performance",
            styles["Heading2"]
        )
    )

    results = [
        ["Parameter", "Value"],
        ["Drag Coefficient (Cd)", f"{predicted_cd:.6f}"],
        ["Lift Coefficient (Cl)", f"{predicted_cl:.6f}"]
    ]

    table = Table(
        results,
        colWidths=[250, 150]
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                8
            )
        ])
    )

    elements.append(table)

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "Vehicle Geometry",
            styles["Heading2"]
        )
    )

    feature_names = [
        "Body Length",
        "Body Height",
        "Body Width",
        "Front Arc Diameter",
        "Slant Angle Length",
        "Slant Angle Height",
        "Slant Surface Length",
        "Slant Angle Degrees"
    ]

    geometry_data = [
        ["Parameter", "Value"]
    ]

    for name, value in zip(
        feature_names,
        geometry
    ):

        geometry_data.append([
            name,
            f"{value:.6f}"
        ])

    geometry_table = Table(
        geometry_data,
        colWidths=[250, 150]
    )

    geometry_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "PADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    elements.append(
        geometry_table
    )

    elements.append(
        Spacer(1, 20)
    )

    elements.append(
        Paragraph(
            "Note: The aerodynamic coefficients are "
            "machine-learning surrogate-model predictions "
            "and should be CFD-validated before engineering "
            "deployment.",
            styles["BodyText"]
        )
    )

    document.build(elements)