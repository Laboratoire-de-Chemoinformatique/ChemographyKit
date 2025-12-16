Utils Module
============

The utils module contains specialized utilities for different types of analysis with GTM models.

Classification Utils
-------------------

.. automodule:: chemographykit.utils.classification
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: chemographykit.utils.classification.get_class_density_matrix

.. autofunction:: chemographykit.utils.classification.class_density_to_table

.. autofunction:: chemographykit.utils.classification.class_prob_from_density

.. autofunction:: chemographykit.utils.classification.get_class_inds

Regression Utils
---------------

.. automodule:: chemographykit.utils.regression
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: chemographykit.utils.regression.get_reg_density_matrix

.. autofunction:: chemographykit.utils.regression.reg_density_to_table

.. autofunction:: chemographykit.utils.regression.norm_reg_density

Density Utils
-------------

.. automodule:: chemographykit.utils.density
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: chemographykit.utils.density.get_density_matrix

.. autofunction:: chemographykit.utils.density.density_to_table

Molecules Utils
--------------

.. automodule:: chemographykit.utils.molecules
   :members:
   :undoc-members:
   :show-inheritance:

Functions
~~~~~~~~~

.. autofunction:: chemographykit.utils.molecules.calculate_latent_coords

Usage Examples
--------------

Classification Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.utils.classification import get_class_density_matrix, class_density_to_table
   import numpy as np

   # Example responsibilities and binary labels
   responsibilities = np.random.rand(1000, 100)  # 1000 molecules, 100 nodes
   class_labels = np.random.choice([0, 1], size=1000)
   class_names = ["Inactive", "Active"]

   # Calculate class density matrices
   density, class_density, class_prob = get_class_density_matrix(
       responsibilities,
       class_labels,
       class_name=class_names,
       normalize=True
   )

   # Create table for visualization
   class_table = class_density_to_table(
       density, class_density, class_prob,
       node_threshold=0.1,
       class_name=class_names,
       normalized=True
   )

   print(f"Classification table shape: {class_table.shape}")

Multi-class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Multi-class example
   class_labels = np.random.choice([0, 1, 2], size=1000)
   class_names = ["Low", "Medium", "High"]

   density, class_density, class_prob = get_class_density_matrix(
       responsibilities,
       class_labels,
       class_name=class_names,
       normalize=True
   )

Regression Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.utils.regression import get_reg_density_matrix, reg_density_to_table

   # Example regression values
   regression_values = np.random.normal(5.0, 2.0, size=1000)

   # Calculate regression density matrix
   density, reg_density = get_reg_density_matrix(responsibilities, regression_values)

   # Create table for visualization
   reg_table = reg_density_to_table(
       density, reg_density,
       node_threshold=0.1
   )

   print(f"Regression table shape: {reg_table.shape}")

Density Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.utils.density import get_density_matrix, density_to_table

   # Calculate density matrix
   density = get_density_matrix(responsibilities)

   # Create density table for visualization
   density_table = density_to_table(density, node_threshold=0.1)

   print(f"Density table shape: {density_table.shape}")

Molecular Coordinates
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.utils.molecules import calculate_latent_coords

   # Calculate molecular coordinates for plotting
   mol_coords = calculate_latent_coords(
       responsibilities,
       correction=True,  # Apply coordinate correction
       return_node=True  # Include most responsible node
   )

   print(mol_coords.head())
   print(f"Columns: {mol_coords.columns.tolist()}")

Advanced Usage
~~~~~~~~~~~~~

.. code-block:: python

   # Complete workflow for classification landscape
   from chemographykit.gtm import GTM
   from chemographykit.utils.classification import get_class_density_matrix, class_density_to_table
   from chemographykit.plots.plotly_landscapes import plotly_discrete_class_landscape
   import torch
   import numpy as np

   # Generate data and labels
   data = torch.randn(1000, 50, dtype=torch.float64)
   labels = np.random.choice([0, 1], size=1000)

   # Train GTM
   gtm = GTM(num_nodes=100, num_basis_functions=25)
   gtm.fit(data)

   # Project data
   responsibilities, _ = gtm.project(data)
   responsibilities_np = responsibilities.T.cpu().numpy()

   # Analyze classification
   density, class_density, class_prob = get_class_density_matrix(
       responsibilities_np,
       labels,
       class_name=["Inactive", "Active"],
       normalize=True
   )

   # Create visualization table
   class_table = class_density_to_table(
       density, class_density, class_prob,
       node_threshold=0.0,
       class_name=["Inactive", "Active"],
       normalized=True
   )

   # Generate landscape plot
   fig = plotly_discrete_class_landscape(
       class_table,
       title="GTM Classification Landscape",
       first_class_label="Inactive",
       second_class_label="Active"
   )

Notes
-----

Data Format Requirements
~~~~~~~~~~~~~~~~~~~~~~~

All utility functions expect:

- **responsibilities**: NumPy array of shape (n_molecules, n_nodes)
- **class_labels**: 1D array of class labels (integers or strings)
- **regression_values**: 1D array of continuous values
- **node_threshold**: Minimum density threshold for visualization

Table Output Format
~~~~~~~~~~~~~~~~~~

The table functions return pandas DataFrames with standardized columns:

- **x, y**: Grid coordinates
- **density**: Normalized density values
- **class_density_[class]**: Density for each class
- **class_prob_[class]**: Probability for each class (classification only)
- **reg_density**: Mean regression value (regression only)

Normalization Options
~~~~~~~~~~~~~~~~~~~~

- **normalize=True**: Normalizes densities to sum to 1.0
- **normalize=False**: Uses raw density counts

The normalization affects how the landscapes are colored and interpreted in visualizations.
