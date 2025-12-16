Plots Module
============

The plots module provides visualization capabilities for GTM landscapes using both interactive (Plotly) and static (Altair) plotting libraries.

Plotly Landscapes
-----------------

Interactive visualizations with smooth interpolation and hover information.

.. automodule:: chemographykit.plots.plotly_landscapes
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: chemographykit.plots.plotly_landscapes.plotly_smooth_density_landscape

.. autofunction:: chemographykit.plots.plotly_landscapes.plotly_discrete_class_landscape

.. autofunction:: chemographykit.plots.plotly_landscapes.plotly_smooth_regression_landscape

Altair Landscapes
-----------------

Static, publication-ready visualizations with discrete grid representation.

.. automodule:: chemographykit.plots.altair_landscapes
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: chemographykit.plots.altair_landscapes.altair_discrete_density_landscape

.. autofunction:: chemographykit.plots.altair_landscapes.altair_discrete_class_landscape

.. autofunction:: chemographykit.plots.altair_landscapes.altair_discrete_regression_landscape

.. autofunction:: chemographykit.plots.altair_landscapes.altair_points_chart

Usage Examples
--------------

Interactive Density Landscape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.plotly_landscapes import plotly_smooth_density_landscape
   from chemographykit.utils.density import get_density_matrix, density_to_table
   import numpy as np

   # Assuming you have responsibilities from GTM projection
   responsibilities = np.random.rand(1000, 100)  # Example data

   # Calculate density
   density = get_density_matrix(responsibilities)
   density_table = density_to_table(density, node_threshold=0.1)

   # Create interactive plot
   fig = plotly_smooth_density_landscape(
       density_table,
       title="GTM Density Landscape",
       node_threshold=0.1,
       width=800,
       height=600
   )
   fig.show()

Interactive Classification Landscape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.plotly_landscapes import plotly_discrete_class_landscape
   from chemographykit.utils.classification import get_class_density_matrix, class_density_to_table

   # Example classification data
   class_labels = np.random.choice([0, 1], size=1000)

   # Calculate class densities
   density, class_density, class_prob = get_class_density_matrix(
       responsibilities,
       class_labels,
       class_name=["Inactive", "Active"],
       normalize=True
   )

   # Create classification table
   class_table = class_density_to_table(
       density, class_density, class_prob,
       node_threshold=0.0,
       class_name=["Inactive", "Active"],
       normalized=True
   )

   # Create interactive plot
   fig = plotly_discrete_class_landscape(
       class_table,
       title="GTM Classification Landscape",
       first_class_label="Inactive",
       second_class_label="Active",
       min_density=0.1,
       width=800,
       height=600
   )
   fig.show()

Interactive Regression Landscape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.plotly_landscapes import plotly_smooth_regression_landscape
   from chemographykit.utils.regression import get_reg_density_matrix, reg_density_to_table

   # Example regression data
   regression_values = np.random.normal(5.0, 2.0, size=1000)

   # Calculate regression densities
   density, reg_density = get_reg_density_matrix(responsibilities, regression_values)

   # Create regression table
   reg_table = reg_density_to_table(
       density, reg_density,
       node_threshold=0.0
   )

   # Create interactive plot
   fig = plotly_smooth_regression_landscape(
       reg_table,
       title="GTM Regression Landscape",
       regression_label="Property Value",
       min_density=0.1,
       colorscale="Viridis",
       width=800,
       height=600
   )
   fig.show()

Static Density Landscape
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.altair_landscapes import altair_discrete_density_landscape

   # Create static density plot
   chart = altair_discrete_density_landscape(
       density_table,
       title="GTM Density Map",
       width=400,
       height=400
   )
   chart.show()

Static Classification Landscape
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.altair_landscapes import altair_discrete_class_landscape

   # Create static classification plot
   chart = altair_discrete_class_landscape(
       class_table,
       title="GTM Classification Map",
       first_class_label="Inactive",
       second_class_label="Active",
       width=400,
       height=400
   )
   chart.show()

Static Regression Landscape
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.altair_landscapes import altair_discrete_regression_landscape

   # Create static regression plot
   chart = altair_discrete_regression_landscape(
       reg_table,
       title="GTM Regression Map",
       regression_label="Property Value",
       width=400,
       height=400
   )
   chart.show()

Overlay Molecular Points
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.plots.altair_landscapes import altair_points_chart
   from chemographykit.utils.molecules import calculate_latent_coords

   # Calculate molecular coordinates
   mol_coords = calculate_latent_coords(
       responsibilities,
       correction=True,
       return_node=True
   )

   # Create points overlay
   points_chart = altair_points_chart(
       mol_coords,
       num_nodes=100,
       points_size=30,
       points_color="red",
       points_opacity=0.7
   )

   # Combine with landscape
   combined = altair_discrete_density_landscape(density_table) + points_chart
   combined.show()

Advanced Customization
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Plotly with custom styling
   fig = plotly_smooth_density_landscape(
       density_table,
       title="Custom GTM Landscape",
       node_threshold=0.05,
       colorscale="Plasma",
       width=1000,
       height=800,
       show_colorbar=True
   )

   # Update layout for publication
   fig.update_layout(
       font=dict(size=14),
       title_font_size=18,
       showlegend=False,
       margin=dict(l=50, r=50, t=80, b=50)
   )

   # Altair with custom styling
   chart = altair_discrete_density_landscape(
       density_table,
       title="Publication Ready GTM Map",
       width=500,
       height=500
   ).resolve_scale(
       color='independent'
   ).configure_title(
       fontSize=16,
       font='Arial'
   ).configure_axis(
       labelFontSize=12,
       titleFontSize=14
   )

Comparison Plots
~~~~~~~~~~~~~~~

.. code-block:: python

   import plotly.graph_objects as go
   from plotly.subplots import make_subplots

   # Create side-by-side comparison
   fig = make_subplots(
       rows=1, cols=2,
       subplot_titles=("Training Set", "Test Set"),
       shared_yaxes=True
   )

   # Add training set landscape
   fig1 = plotly_smooth_density_landscape(train_density_table, title="")
   fig.add_trace(fig1.data[0], row=1, col=1)

   # Add test set landscape
   fig2 = plotly_smooth_density_landscape(test_density_table, title="")
   fig.add_trace(fig2.data[0], row=1, col=2)

   fig.update_layout(title="Training vs Test Set Comparison")
   fig.show()

Notes
-----

Plotly vs Altair
~~~~~~~~~~~~~~~~

**Plotly Advantages:**
- Interactive hover information
- Smooth interpolation between grid points
- Zooming and panning capabilities
- Easy to customize colors and styling
- Better for exploratory data analysis

**Altair Advantages:**
- Publication-ready static plots
- Precise discrete grid representation
- Better for overlaying multiple chart types
- Consistent styling across plots
- Smaller file sizes for sharing

Color Scales
~~~~~~~~~~~

Available color scales for Plotly:
- "Viridis", "Plasma", "Inferno", "Magma"
- "Blues", "Greens", "Reds", "YlOrRd"
- "RdYlBu", "Spectral", "Jet"

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~

- **Large datasets**: Use higher node_threshold to reduce plot complexity
- **Interactive plots**: Consider reducing grid resolution for better performance
- **Static plots**: Altair handles large datasets more efficiently
- **File sizes**: Static plots are smaller for sharing and embedding

Customization Tips
~~~~~~~~~~~~~~~~~

- Use consistent color schemes across related plots
- Adjust node_threshold based on data density
- Consider your audience (interactive for exploration, static for publication)
- Test different color scales for accessibility
- Use appropriate titles and labels for clarity
