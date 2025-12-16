Quick Start Guide
================

This guide will get you up and running with ChemographyKit in just a few minutes.

Basic GTM Training
------------------

Let's start with a simple example of training a GTM model:

.. code-block:: python

   import torch
   import numpy as np
   from chemographykit.gtm import GTM

   # Generate sample data (1000 samples, 50 features)
   data = torch.randn(1000, 50, dtype=torch.float64)

   # Create GTM model
   gtm = GTM(
       num_nodes=100,           # 10x10 grid in latent space
       num_basis_functions=25,  # 5x5 RBF centers
       basis_width=0.3,         # RBF width parameter
       reg_coeff=0.01,          # Regularization coefficient
       device="cuda"            # Use GPU if available
   )

   # Fit model and transform data
   latent_coords = gtm.fit_transform(data)
   print(f"Latent coordinates shape: {latent_coords.shape}")  # (2, 1000)

   # Get responsibilities for landscape analysis
   responsibilities, log_likelihoods = gtm.project(data)
   print(f"Responsibilities shape: {responsibilities.shape}")  # (100, 1000)

Creating Visualizations
----------------------

Density Landscapes
~~~~~~~~~~~~~~~~~~

Create density landscapes to visualize data distribution:

.. code-block:: python

   import numpy as np
   from chemographykit.utils.density import get_density_matrix, density_to_table
   from chemographykit.plots.plotly_landscapes import plotly_smooth_density_landscape

   # Calculate density matrix
   responsibilities_np = responsibilities.T.cpu().numpy()
   density = get_density_matrix(responsibilities_np)

   # Create density table for visualization
   density_table = density_to_table(density, node_threshold=0.1)

   # Generate interactive Plotly landscape
   fig = plotly_smooth_density_landscape(
       density_table,
       title="GTM Density Landscape",
       node_threshold=0.1
   )
   fig.show()

Classification Landscapes
~~~~~~~~~~~~~~~~~~~~~~~~

Visualize classification patterns:

.. code-block:: python

   from chemographykit.utils.classification import get_class_density_matrix, class_density_to_table
   from chemographykit.plots.plotly_landscapes import plotly_discrete_class_landscape

   # Sample binary classification labels
   class_labels = np.random.choice([0, 1], size=1000)
   class_names = ["Inactive", "Active"]

   # Calculate class density matrices
   density, class_density, class_prob = get_class_density_matrix(
       responsibilities_np,
       class_labels,
       class_name=class_names,
       normalize=True
   )

   # Create classification table
   class_table = class_density_to_table(
       density, class_density, class_prob,
       node_threshold=0.0,
       class_name=class_names,
       normalized=True
   )

   # Generate classification landscape
   fig = plotly_discrete_class_landscape(
       class_table,
       title="GTM Classification Landscape",
       first_class_label="Inactive",
       second_class_label="Active",
       min_density=0.1
   )
   fig.show()

Regression Landscapes
~~~~~~~~~~~~~~~~~~~~

Visualize continuous properties:

.. code-block:: python

   from chemographykit.utils.regression import get_reg_density_matrix, reg_density_to_table
   from chemographykit.plots.plotly_landscapes import plotly_smooth_regression_landscape

   # Sample regression values
   regression_values = np.random.normal(5.0, 2.0, size=1000)

   # Calculate regression density matrix
   density, reg_density = get_reg_density_matrix(responsibilities_np, regression_values)

   # Create regression table
   reg_table = reg_density_to_table(
       density, reg_density,
       node_threshold=0.0
   )

   # Generate regression landscape
   fig = plotly_smooth_regression_landscape(
       reg_table,
       title="GTM Regression Landscape",
       regression_label="Property Value",
       min_density=0.1
   )
   fig.show()

Advanced Features
----------------

Responsibility Pattern (RP) Fingerprints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ChemographyKit provides unique fingerprints based on responsibility patterns:

.. code-block:: python

   from chemographykit.metrics import resp_to_pattern, compute_rp_coverage

   # Convert responsibilities to RP fingerprints
   rp_fingerprints = np.array([
       resp_to_pattern(resp, n_bins=10, threshold=0.01)
       for resp in responsibilities_np
   ])

   # Calculate coverage between datasets
   reference_fps = rp_fingerprints[:500]  # First 500 as reference
   test_fps = rp_fingerprints[500:]       # Last 500 as test

   coverage = compute_rp_coverage(reference_fps, test_fps, use_weight=True)
   print(f"Weighted coverage: {coverage:.3f}")

Molecular Coordinates
~~~~~~~~~~~~~~~~~~~~

Calculate molecular coordinates for plotting:

.. code-block:: python

   from chemographykit.utils.molecules import calculate_latent_coords

   # Calculate molecular coordinates
   mol_coords = calculate_latent_coords(
       responsibilities_np,
       correction=True,  # Adjust for visualization
       return_node=True  # Include most responsible node
   )
   print(mol_coords.head())

Static Visualizations with Altair
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create publication-ready static plots:

.. code-block:: python

   from chemographykit.plots.altair_landscapes import (
       altair_discrete_density_landscape,
       altair_discrete_class_landscape,
       altair_points_chart
   )

   # Create discrete density landscape
   density_chart = altair_discrete_density_landscape(
       density_table,
       title="GTM Density Map"
   )

   # Overlay molecular points
   points_chart = altair_points_chart(
       mol_coords,
       num_nodes=100,
       points_size=50,
       points_color="red"
   )

   # Combine charts
   combined = density_chart + points_chart
   combined.show()

Model Configuration
------------------

GTM Parameters
~~~~~~~~~~~~~

Key parameters for GTM configuration:

* **num_nodes**: Number of latent space grid nodes (must be perfect square for 2D)
* **num_basis_functions**: Number of RBF centers (must be perfect square for 2D)
* **basis_width**: RBF width parameter (controls smoothness)
* **reg_coeff**: Regularization coefficient (prevents overfitting)
* **standardize**: Whether to standardize input data (recommended: True)
* **max_iter**: Maximum EM algorithm iterations
* **tolerance**: Convergence tolerance
* **device**: Computation device ("cpu" or "cuda")

Example with custom parameters:

.. code-block:: python

   gtm = GTM(
       num_nodes=144,           # 12x12 grid
       num_basis_functions=36,  # 6x6 RBF centers
       basis_width=0.5,         # Wider RBFs
       reg_coeff=0.001,         # Less regularization
       standardize=True,        # Standardize data
       max_iter=200,           # More iterations
       tolerance=1e-6,         # Tighter convergence
       device="cuda"           # Use GPU
   )

Performance Tips
---------------

1. **Use GPU**: Set ``device="cuda"`` for significant speedup on large datasets
2. **Choose appropriate grid size**: Balance between resolution and computational cost
3. **PCA initialization**: Use ``GTM`` class instead of ``VanillaGTM`` for better convergence
4. **Data standardization**: Always enable for numerical stability
5. **Batch processing**: Process large datasets in chunks if memory is limited

Next Steps
----------

* Explore the :doc:`tutorials/index` for detailed examples
* Check the :doc:`api/gtm` for complete API reference
* See :doc:`examples/index` for real-world applications
