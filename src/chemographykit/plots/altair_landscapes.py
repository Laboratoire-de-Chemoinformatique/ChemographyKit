from typing import List

import altair as alt
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

from chemographykit.utils.density import calculate_grid


def _infer_vega_type(series: pd.Series) -> str:
    """
    Infer a sensible Vega-Lite type code for Altair: 'Q', 'O', 'N', or 'T'.
    - Numbers (except bool) -> 'Q'
    - datetimes -> 'T'
    - categoricals -> 'O'
    - everything else -> 'N'
    """
    if is_bool_dtype(series):
        # Bool behaves better as nominal (two categories) than quantitative
        return "N"
    if is_datetime64_any_dtype(series):
        return "T"
    if is_numeric_dtype(series):
        return "Q"
    if is_categorical_dtype(series):
        # ordered categoricals are conceptually ordinal; nominal is also ok
        return "O" if getattr(series.dtype, "ordered", False) else "N"
    return "N"

def altair_points_chart(
    points_table: pd.DataFrame, 
    num_nodes: int, 
    points_size: int = 100, 
    points_opacity: float = 1,
    points_color: str = "#228be6",
    coloring_column: str = None,
    coloring_scheme: str = "set1",
    tooltip_columns: dict = None,
    legend=None,
    title="",
):
    axis_len = int(np.sqrt(num_nodes))
    legend_config = None if legend is None else legend

    if coloring_column is not None:
        color = alt.Color( # analogue to z axis in Plotly
            f"{coloring_column}:O",
            legend=legend_config,
            scale=alt.Scale(scheme=alt.SchemeParams(name=coloring_scheme)), 
        )
         # here we can specify color schemes, check https://vega.github.io/vega/docs/schemes/
    else:
        color = alt.Color()

    tooltip = []
    if tooltip_columns is not None:
        for name, title_name in tooltip_columns.items():
            if name == "image":
                tooltip.append(alt.Tooltip("image"))
            else:
                tooltip.append(alt.Tooltip(name, title=title_name))
   
    points = alt.Chart(points_table, title=title).mark_circle(
        size=points_size,
        opacity=points_opacity,
        color=points_color
    ).encode(
        x=alt.X(  # we have to keep them almost the same as in density chart
            "x:Q",
            title=None, 
            axis=None, 
            scale=alt.Scale(domain=[1, axis_len + 1]) # here we have to specify min and max, as for points we have quantatitive type (ordinal will fail)
        ),
        y=alt.Y(
            "y:Q", 
            title=None, 
            axis=None, 
            scale=alt.Scale(domain=[1, axis_len + 1], reverse=True)
        ),
        color=color,
        tooltip=tooltip,
    )
    return points


def altair_discrete_density_landscape(density_table, title=""):
    """
    It takes a density vector and returns an Altair-based visualisation of GTMap

    :param density: the density of the nodes
    :param node_threshold: minimal density, where values lower than specified means that the node is empty
    :return: A chart object
    """
    n_nodes = density_table.shape[0]
    axis_len = int(np.sqrt(n_nodes))

    chart = (
        alt.Chart(density_table, title=title)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            color=alt.Color(
                "filtered_density:Q",
                title=None,
                scale=alt.Scale(scheme=alt.SchemeParams(name="greys")),
            ),
            tooltip=[
                alt.Tooltip("nodes", title="Node"),
                alt.Tooltip("density", title="Density"),
            ],
        )
    )

    return chart


def altair_discrete_class_landscape(
    class_density_table,
    colorset="lighttealblue",
    title="",
    use_density=False,
    first_class_prob_column_name="first_class_prob",
    second_class_prob_column_name="second_class_prob",
    first_class_density_column_name="first_class_density",
    second_class_density_column_name="second_class_density",
    first_class_label="Inactive",
    second_class_label="Active",
    reverse=False,
):
    n_nodes = class_density_table.shape[0]
    axis_len = int(np.sqrt(n_nodes))

    opacity = alt.Opacity()
    if use_density:
        opacity = alt.Opacity("density", title=None, legend=None)

    tooltip = [
        alt.Tooltip("nodes", title="Node"),
        alt.Tooltip("density", title="Density"),
        alt.Tooltip(first_class_prob_column_name, title=first_class_label + " prob"),
        alt.Tooltip(second_class_prob_column_name, title=second_class_label + " prob"),
        alt.Tooltip(
            first_class_density_column_name, title=first_class_label + " density"
        ),
        alt.Tooltip(
            second_class_density_column_name, title=second_class_label + " density"
        ),
    ]

    chart = (
        alt.Chart(
            class_density_table,
            title=title,
        )
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            color=alt.Color(
                f"{second_class_prob_column_name}:Q",
                title=None,
                scale=alt.Scale(
                    domain=[0, 1],
                    scheme=alt.SchemeParams(name=colorset),
                    reverse=reverse,
                ),
                legend=alt.Legend(),
            ),
            tooltip=tooltip,
            opacity=opacity,
        )
    )
    return chart


def altair_discrete_regression_landscape(
    reg_density_table,
    colorset="lighttealblue",
    use_density=False,
    scale_type="linear",
    regval_domain=None,
    reverse=False,
    title="",
    regression_label="Regression value"
):
    # --- minimal fix: derive grid size from table coords, not row count ---
    # works even if reg_density_table was filtered by node_threshold
    axis_len = int(max(reg_density_table["x"].max(), reg_density_table["y"].max()))

    opacity = alt.Opacity()
    if use_density:
        opacity = alt.Opacity("density", title=None, legend=None)

    tooltip = [
        alt.Tooltip("nodes", title="Node"),
        alt.Tooltip("density", title="Density"),
        alt.Tooltip("filtered_reg_density", title=regression_label),
    ]

    if regval_domain:
        color_scale = alt.Scale(
            domain=regval_domain,
            scheme=alt.SchemeParams(name=colorset),
            type=scale_type,
            reverse=reverse,
        )
    else:
        color_scale = alt.Scale(
            scheme=alt.SchemeParams(name=colorset),
            type=scale_type,
            reverse=reverse,
        )

    chart = (
        alt.Chart(reg_density_table, title=title)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            color=alt.Color(
                "filtered_reg_density:Q",
                title=None,
                scale=color_scale,
                legend=alt.Legend(),
            ),
            tooltip=tooltip,
            opacity=opacity,
        )
    )
    return chart


def altair_discrete_query_landscape(
    class_density_table,
    colorset="viridis",
    title="",
    criteria_column="criteria_satisfied",
    reverse=True,
):
    """
    Generate landscape using Altair to visualize discrete criteria values over a 2D layout of nodes.
    Colors are applied only to cells with non-null criteria values.

    Parameters:
        class_density_table (pd.DataFrame): A DataFrame containing at least the following columns:
            - 'x': X grid coordinate (categorical or ordered)
            - 'y': Y grid coordinate (categorical or ordered)
            - 'nodes': Node IDs
            - 'density': A numeric value to include in tooltips
            - criteria_column (str): A column with discrete criteria values (can include NaN)

        colorset (str): Name of the color scheme to use (e.g., 'viridis', 'category10').
        title (str): Optional title to display above the chart.
        criteria_column (str): Column used for determining coloring of the cells.
        reverse (bool): Whether to reverse the color scale (default is True).

    Returns:
        alt.Chart: An Altair chart object representing the discrete query landscape.
    """
    n_nodes = class_density_table.shape[0]
    axis_len = int(np.sqrt(n_nodes))

    non_null_values = class_density_table[criteria_column].dropna().unique().tolist()
    non_null_values = sorted(non_null_values)

    tooltip = [
        alt.Tooltip("nodes:O", title="Node"),
        alt.Tooltip("density", title="Density"),
        alt.Tooltip(f"{criteria_column}:N", title="Criteria Satisfied"),
    ]

    chart = (
        alt.Chart(class_density_table, title=title)
        .mark_rect()
        .encode(
            x=alt.X(
                "x:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            y=alt.Y(
                "y:O",
                title=None,
                axis=None,
                scale=alt.Scale(domain=list(range(1, axis_len + 1))),
            ),
            color=alt.condition(
                f"datum['{criteria_column}'] !== null",
                alt.Color(
                    f"{criteria_column}:N",
                    title="Criteria Satisfied",
                    scale=alt.Scale(
                        scheme=colorset, domain=non_null_values, reverse=reverse
                    ),
                    legend=alt.Legend(),
                ),
                alt.value("white"),
            ),
            tooltip=tooltip,
        )
    )
    return chart
