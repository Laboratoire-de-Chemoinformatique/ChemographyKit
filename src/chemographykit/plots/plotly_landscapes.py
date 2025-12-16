import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plotly_smooth_density_landscape(
    density_table,
    colorset=None,
    title="Density landscape",
    node_threshold=0.1,
    width=800,
    height=800,
    background_color="white",
    use_smooth=True,
):
    """
    Smoothed overall density heatmap with masking of low-density nodes.
    Always uses the raw 'density' column and applies the threshold inside this function,
    ensuring consistency regardless of any prior filtering.

    Parameters
    ----------
    density_table : pd.DataFrame
        Must contain at least:
          - 'x', 'y'   : grid coordinates
          - 'density'  : raw density value per node
          - 'nodes'    : node index (for hover text)
    colorset : list or None
        Plotly colorscale; if None a default GTM rainbow is used.
    title : str
        Title of the figure.
    node_threshold : float
        Nodes with density below this are masked out (shown white).
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.
    background_color : str
        Paper background color.
    use_smooth : bool
        Whether to enable Plotly’s `zsmooth="best"` smoothing.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    # 1) Pivot raw density and node IDs
    pivot_density = density_table.pivot(
        index="y", columns="x", values="density"
    ).sort_index(ascending=False)
    pivot_nodes = density_table.pivot(
        index="y", columns="x", values="nodes"
    ).sort_index(ascending=False)
    z_raw = pivot_density.values.astype(float)
    nodes = pivot_nodes.values
    xvals = pivot_density.columns.tolist()
    yvals = pivot_density.index.tolist()

    # 2) Build mask from raw density
    mask = z_raw >= node_threshold

    # 3) Build hover‐text matrix
    rows, cols = z_raw.shape
    text = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                text[i, j] = f"Node: {nodes[i,j]}<br>No density"
            else:
                text[i, j] = f"Node: {nodes[i,j]}<br>Density: {z_raw[i,j]:.2f}"

    # 4) Default GTM rainbow if none supplied
    if colorset is None:
        colorset = [
            [0.0, "rgb(0,0,153)"],
            [0.1, "rgb(0,0,153)"],
            [0.1, "rgb(0,102,204)"],
            [0.2, "rgb(0,102,204)"],
            [0.2, "rgb(51,153,255)"],
            [0.3, "rgb(51,153,255)"],
            [0.3, "rgb(0,255,245)"],
            [0.4, "rgb(0,255,245)"],
            [0.4, "rgb(0,204,0)"],
            [0.5, "rgb(0,204,0)"],
            [0.5, "rgb(102,255,51)"],
            [0.6, "rgb(102,255,51)"],
            [0.6, "rgb(255,255,0)"],
            [0.7, "rgb(255,255,0)"],
            [0.7, "rgb(255,175,107)"],
            [0.8, "rgb(255,175,107)"],
            [0.8, "rgb(255,128,0)"],
            [0.9, "rgb(255,128,0)"],
            [0.9, "rgb(255,0,0)"],
            [1.0, "rgb(255,0,0)"],
        ]

    # 5) Background heatmap: full raw density
    bg = go.Heatmap(
        z=z_raw,
        x=xvals,
        y=yvals,
        colorscale=colorset,
        zmin=0,
        zmax=np.nanmax(z_raw),
        text=text,
        hoverinfo="text",
        showscale=True,
        colorbar=dict(title="Density"),
    )
    if use_smooth:
        bg.update(zsmooth="best")

    # 6) White‐mask overlay for below‐threshold
    max_den = np.nanmax(z_raw) if np.nanmax(z_raw) != 0 else 1.0
    dens_norm = z_raw / max_den
    thr_norm = node_threshold / max_den
    mask_trace = go.Heatmap(
        z=dens_norm,
        x=xvals,
        y=yvals,
        colorscale=[
            [0.0, "rgba(255,255,255,1)"],
            [thr_norm, "rgba(255,255,255,1)"],
            [thr_norm + 1e-6, "rgba(255,255,255,0)"],
            [1.0, "rgba(255,255,255,0)"],
        ],
        hoverinfo="none",
        showscale=False,
        zmin=0,
        zmax=1,
    )
    if use_smooth:
        mask_trace.update(zsmooth="best")

    # 7) Invisible hover layer
    hover = go.Heatmap(
        z=np.zeros_like(z_raw),
        x=xvals,
        y=yvals,
        text=text,
        hoverinfo="text",
        showscale=False,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        zmin=0,
        zmax=1,
    )

    # 8) Assemble & return
    fig = go.Figure(data=[bg, mask_trace, hover])
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def plotly_discrete_class_landscape(
    class_density_table,
    colorset=None,
    title="Classification landscape",
    first_class_prob_column_name="1_norm_prob",
    second_class_prob_column_name="2_norm_prob",
    first_class_density_column_name="1_norm_density",
    second_class_density_column_name="2_norm_density",
    density_column_name="density",
    node_column_name="nodes",
    first_class_label="Inactive",
    second_class_label="Active",
    min_density=0.1,
    width=800,
    height=800,
    background_color="white",
    use_smooth=True,
):
    """
    Smoothed classification heatmap with masking of low-density nodes.
    Be carefull, node threshold used to create the class_density_table must be set to 0 or there might be errors.

    Parameters
    ----------
    class_density_table : pd.DataFrame
        Must contain at least:
          - 'x', 'y' : grid coordinates of each node
          - density_column_name : overall node density
          - second_class_prob_column_name : probability of the second class
        Optionally may also contain:
          - first_class_prob_column_name  : probability of the first class
          - first_class_density_column_name, second_class_density_column_name
               : per-class densities
          - node_column_name : node index for hover text

    colorset : list or None
        A Plotly colorscale (list of [t, color] pairs). If None, defaults to the GTM rainbow.

    title : str
        Title of the figure.

    first_class_prob_column_name : str
        Column name holding the first-class probability in `class_density_table`.

    second_class_prob_column_name : str
        Column name holding the second-class probability.

    first_class_density_column_name : str
        Column name holding the first-class density.

    second_class_density_column_name : str
        Column name holding the second-class density.

    density_column_name : str
        Column name holding the overall (summed) node density.

    node_column_name : str
        Column name holding node indices, used in hover tooltips.

    min_density : float
        Nodes with overall density below this threshold will be masked (shown as white).
        Must be ≥ the threshold that was used when generating `class_density_table`.

    width : int
        Width of the figure in pixels.

    height : int
        Height of the figure in pixels.

    background_color : str
        Background color for both plot and paper (e.g. `"white"` or `"rgba(0,0,0,0)"`).

    use_smooth : bool
        Whether to enable Plotly’s smoothing (`zsmooth="best"`) on the probability layer.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        An overlaid figure with three layers:
          1. Background: smoothed second-class probability heatmap.
          2. Mask: solid white overlay where density < min_density.
          3. Hover: invisible transparent layer carrying full tooltips.
    """

    # 1) Pivot Z and density grids
    prob_p = class_density_table.pivot(
        index="y", columns="x", values=second_class_prob_column_name
    ).sort_index(ascending=False)
    dens_p = class_density_table.pivot(
        index="y", columns="x", values=density_column_name
    ).sort_index(ascending=False)
    raw_z = prob_p.values.astype(float)
    dens = dens_p.values.astype(float)
    xvals, yvals = prob_p.columns.tolist(), prob_p.index.tolist()

    # 2) Pivot all tooltip columns
    tooltip_cols = []
    for col, name in [
        (node_column_name, "Node"),
        (density_column_name, "Density"),
        (first_class_prob_column_name, f"{first_class_label} Prob"),
        (second_class_prob_column_name, f"{second_class_label} Prob"),
        (first_class_density_column_name, f"{first_class_label} Density"),
        (second_class_density_column_name, f"{second_class_label} Density"),
    ]:
        if col in class_density_table.columns:
            mat = (
                class_density_table.pivot(index="y", columns="x", values=col)
                .sort_index(ascending=False)
                .values
            )
            tooltip_cols.append((mat, name))

    # 3) Build the 2D text array for hover
    rows, cols = raw_z.shape
    mask = dens < min_density
    text = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            if mask[i, j]:
                # under threshold
                if node_column_name in class_density_table.columns:
                    node_val = tooltip_cols[0][0][i, j]  # Node is first
                    text[i, j] = f"Node: {node_val}<br>No density"
                else:
                    text[i, j] = "No density"
            else:
                lines = []
                for mat, label in tooltip_cols:
                    val = mat[i, j]
                    if isinstance(val, float):
                        val_str = f"{val:.2f}"
                    else:
                        val_str = str(val)
                    lines.append(f"{label}: {val_str}")
                text[i, j] = "<br>".join(lines)

    # 4) Default colorscale
    if colorset is None:
        colorset = [
            [0.0, "rgb(0,0,153)"],
            [0.1, "rgb(0,0,153)"],
            [0.1, "rgb(0,102,204)"],
            [0.2, "rgb(0,102,204)"],
            [0.2, "rgb(51,153,255)"],
            [0.3, "rgb(51,153,255)"],
            [0.3, "rgb(0,255,245)"],
            [0.4, "rgb(0,255,245)"],
            [0.4, "rgb(0,204,0)"],
            [0.5, "rgb(0,204,0)"],
            [0.5, "rgb(102,255,51)"],
            [0.6, "rgb(102,255,51)"],
            [0.6, "rgb(255,255,0)"],
            [0.7, "rgb(255,255,0)"],
            [0.7, "rgb(255,175,107)"],
            [0.8, "rgb(255,175,107)"],
            [0.8, "rgb(255,128,0)"],
            [0.9, "rgb(255,128,0)"],
            [0.9, "rgb(255,0,0)"],
            [1.0, "rgb(255,0,0)"],
        ]

    # 5) Background heatmap (smoothed)
    bg = go.Heatmap(
        z=raw_z,
        x=xvals,
        y=yvals,
        colorscale=colorset,
        zmin=0,
        zmax=1,
        text=text,
        hoverinfo="text",
        showscale=True,
        colorbar=dict(title="Prob"),
    )
    if use_smooth:
        bg.update(zsmooth="best")

    # 6) White mask as second trace
    dens_norm = dens / dens.max()
    thr = min_density / dens.max()
    mask_trace = go.Heatmap(
        z=dens_norm,
        x=xvals,
        y=yvals,
        colorscale=[
            [0.0, "rgba(255,255,255,1)"],
            [thr, "rgba(255,255,255,1)"],
            [thr + 1e-6, "rgba(255,255,255,0)"],
            [1.0, "rgba(255,255,255,0)"],
        ],
        hoverinfo="none",
        showscale=False,
        zmin=0,
        zmax=1,
    )
    if use_smooth:
        mask_trace.update(zsmooth="best")

    # 7) Invisible hover layer on top
    hover_trace = go.Heatmap(
        z=np.zeros_like(raw_z),
        x=xvals,
        y=yvals,
        text=text,
        hoverinfo="text",
        showscale=False,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        zmin=0,
        zmax=1,
    )

    # 8) Compose and return
    fig = go.Figure(data=[bg, mask_trace, hover_trace])
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def plotly_smooth_regression_landscape(
    regression_table,
    colorset=None,
    title="",
    width=800,
    height=800,
    regression_label="Regression",
    legend_label="Regression density",
    default_val=None,
    min_density=0.1,
    background_color="white",
    use_smooth=True,
):
    """
    Smoothed regression heatmap with masking of low-density nodes.
    Be carefull, node threshold used to create the regression_table must be set to 0 or there might be errors.

    Parameters
    ----------
    regression_table : pd.DataFrame
        Must contain 'x','y','filtered_reg_density' and optionally 'density' and 'nodes'.
    colorset : list or None
        Plotly colorscale; if None a default GTM rainbow is used.
    title : str
        Title of the figure.
    node_threshold : float
        (Unused here; kept for interface parity.)
    width, height : int
        Figure dimensions.
    regression_label : str
        Label shown in tooltip for the regression value.
    legend_label : str
        Colorbar title.
    default_val : float or None
        Minimum for the color scale. If None inferred from data.
    min_density : float
        Threshold below which density is considered too low and masked.
    background_color : str
        Background color.
    use_smooth : bool
        Whether to enable Plotly smoothing.
    """
    if "filtered_reg_density" not in regression_table.columns:
        raise ValueError("Missing required column: 'filtered_reg_density'.")

    # Pivot required matrices
    pivot_reg = regression_table.pivot(
        index="y", columns="x", values="filtered_reg_density"
    ).sort_index(ascending=False)
    z_raw = pivot_reg.values.astype(float)
    xvals = pivot_reg.columns.tolist()
    yvals = pivot_reg.index.tolist()

    # Optional pivots
    has_density = "density" in regression_table.columns
    if has_density:
        pivot_den = regression_table.pivot(
            index="y", columns="x", values="density"
        ).sort_index(ascending=False)
        dens = pivot_den.values.astype(float)
        mask = dens >= min_density
    else:
        mask = np.ones_like(z_raw, dtype=bool)
        dens = None  # for tooltip logic

    has_nodes = "nodes" in regression_table.columns
    if has_nodes:
        pivot_nodes = regression_table.pivot(
            index="y", columns="x", values="nodes"
        ).sort_index(ascending=False)
        nodes = pivot_nodes.values
    else:
        nodes = None

    # Build tooltip text
    rows, cols = z_raw.shape
    text = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                if has_nodes:
                    text[i, j] = f"Node: {nodes[i,j]}<br>No density"
                else:
                    text[i, j] = "No density"
            else:
                parts = []
                if has_nodes:
                    parts.append(f"Node: {nodes[i,j]}")
                if has_density:
                    parts.append(f"Density: {dens[i,j]:.2f}")
                parts.append(f"{regression_label}: {z_raw[i,j]:.2f}")
                text[i, j] = "<br>".join(parts)

    # Default colorset (GTM rainbow) if not provided
    if colorset is None:
        colorset = [
            [0.0, "rgb(0,0,153)"],
            [0.1, "rgb(0,0,153)"],
            [0.1, "rgb(0,102,204)"],
            [0.2, "rgb(0,102,204)"],
            [0.2, "rgb(51,153,255)"],
            [0.3, "rgb(51,153,255)"],
            [0.3, "rgb(0,255,245)"],
            [0.4, "rgb(0,255,245)"],
            [0.4, "rgb(0,204,0)"],
            [0.5, "rgb(0,204,0)"],
            [0.5, "rgb(102,255,51)"],
            [0.6, "rgb(102,255,51)"],
            [0.6, "rgb(255,255,0)"],
            [0.7, "rgb(255,255,0)"],
            [0.7, "rgb(255,175,107)"],
            [0.8, "rgb(255,175,107)"],
            [0.8, "rgb(255,128,0)"],
            [0.9, "rgb(255,128,0)"],
            [0.9, "rgb(255,0,0)"],
            [1.0, "rgb(255,0,0)"],
        ]

    # Determine zmin/zmax
    visible_vals = np.where(mask, z_raw, np.nan)
    zmin = default_val if default_val is not None else np.nanmin(visible_vals)
    zmax = np.nanmax(visible_vals)

    # Background heatmap
    bg = go.Heatmap(
        z=z_raw,
        x=xvals,
        y=yvals,
        colorscale=colorset,
        zmin=zmin,
        zmax=zmax,
        text=text,
        hoverinfo="text",
        showscale=True,
        colorbar=dict(title=legend_label),
    )
    if use_smooth:
        bg.update(zsmooth="best")

    # White mask overlay for low-density
    if has_density:
        max_den = np.nanmax(dens) if np.nanmax(dens) != 0 else 1.0
        dens_norm = dens / max_den
        thr = min_density / max_den
        mask_trace = go.Heatmap(
            z=dens_norm,
            x=xvals,
            y=yvals,
            colorscale=[
                [0.0, "rgba(255,255,255,1)"],
                [thr, "rgba(255,255,255,1)"],
                [thr + 1e-6, "rgba(255,255,255,0)"],
                [1.0, "rgba(255,255,255,0)"],
            ],
            hoverinfo="none",
            showscale=False,
            zmin=0,
            zmax=1,
        )
        if use_smooth:
            mask_trace.update(zsmooth="best")
    else:
        mask_trace = None

    # Invisible hover layer
    hover = go.Heatmap(
        z=np.zeros_like(z_raw),
        x=xvals,
        y=yvals,
        text=text,
        hoverinfo="text",
        showscale=False,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        zmin=0,
        zmax=1,
    )

    data_traces = [bg]
    if mask_trace is not None:
        data_traces.append(mask_trace)
    data_traces.append(hover)

    fig = go.Figure(data=data_traces)
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        plot_bgcolor=background_color,
        paper_bgcolor=background_color,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )

    return fig
