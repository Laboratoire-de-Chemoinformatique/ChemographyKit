Basic GTM Training
==================

This tutorial covers the fundamentals of training Generative Topographic Mapping (GTM) models using ChemographyKit. You'll learn about different GTM variants, initialization strategies, and basic usage patterns.

Overview
--------

GTM is a probabilistic dimensionality reduction technique that maps high-dimensional data to a lower-dimensional latent space using a generative model. ChemographyKit provides two main implementations:

- **VanillaGTM**: Basic implementation with random initialization
- **GTM**: Enhanced version with PCA-based initialization for better convergence

Prerequisites
-------------

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from chemographykit.gtm import VanillaGTM, GTM

Generating Sample Data
---------------------

Let's start by creating some synthetic data to work with:

.. code-block:: python

   # Set random seed for reproducibility
   torch.manual_seed(42)
   np.random.seed(42)

   # Generate synthetic data with some structure
   n_samples = 1000
   n_features = 50

   # Create data with two clusters
   cluster1 = torch.randn(n_samples // 2, n_features, dtype=torch.float64) + 2
   cluster2 = torch.randn(n_samples // 2, n_features, dtype=torch.float64) - 2
   data = torch.cat([cluster1, cluster2], dim=0)

   print(f"Data shape: {data.shape}")
   print(f"Data type: {data.dtype}")

Basic GTM Training
-----------------

VanillaGTM with Random Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create VanillaGTM model
   vanilla_gtm = VanillaGTM(
       num_nodes=100,           # 10x10 grid in latent space
       num_basis_functions=25,  # 5x5 RBF centers
       basis_width=0.3,         # RBF width parameter
       reg_coeff=0.01,          # Regularization coefficient
       standardize=True,        # Standardize input data
       max_iter=100,           # Maximum EM iterations
       tolerance=1e-5,         # Convergence tolerance
       device="cuda" if torch.cuda.is_available() else "cpu"
   )

   print(f"Model device: {vanilla_gtm.device}")
   print(f"Number of nodes: {vanilla_gtm.num_nodes}")
   print(f"Number of basis functions: {vanilla_gtm.num_basis_functions}")

   # Train the model
   vanilla_gtm.fit(data)

   # Transform data to latent space
   latent_coords = vanilla_gtm.transform(data)
   print(f"Latent coordinates shape: {latent_coords.shape}")

GTM with PCA Initialization
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create GTM model with PCA initialization
   gtm = GTM(
       num_nodes=100,
       num_basis_functions=25,
       basis_width=0.3,
       reg_coeff=0.01,
       standardize=True,
       pca_engine="torch",      # Use PyTorch PCA
       pca_scale=True,          # Scale eigenvectors
       device="cuda" if torch.cuda.is_available() else "cpu"
   )

   # Train the model (PCA initialization happens automatically)
   gtm.fit(data)

   # Transform data
   latent_coords_pca = gtm.transform(data)

Comparing Initialization Methods
-------------------------------

Let's compare the convergence of both methods:

.. code-block:: python

   # Get training history (log-likelihood values)
   vanilla_history = vanilla_gtm.training_history
   pca_history = gtm.training_history

   # Plot convergence
   plt.figure(figsize=(10, 6))
   plt.plot(vanilla_history, label='VanillaGTM (Random Init)', marker='o')
   plt.plot(pca_history, label='GTM (PCA Init)', marker='s')
   plt.xlabel('Iteration')
   plt.ylabel('Log-Likelihood')
   plt.title('GTM Training Convergence')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

   print(f"VanillaGTM final log-likelihood: {vanilla_history[-1]:.2f}")
   print(f"GTM final log-likelihood: {pca_history[-1]:.2f}")

Projecting Data
---------------

Once trained, you can project data to get responsibilities and latent coordinates:

.. code-block:: python

   # Project data to get responsibilities and log-likelihoods
   responsibilities, log_likelihoods = gtm.project(data)

   print(f"Responsibilities shape: {responsibilities.shape}")  # (num_nodes, num_samples)
   print(f"Log-likelihoods shape: {log_likelihoods.shape}")   # (num_samples,)

   # Convert to numpy for further analysis
   responsibilities_np = responsibilities.T.cpu().numpy()  # (num_samples, num_nodes)
   log_likelihoods_np = log_likelihoods.cpu().numpy()

   print(f"Sample responsibility sum: {responsibilities_np[0].sum():.6f}")  # Should be ~1.0

Visualizing Results
------------------

Let's visualize the latent space coordinates:

.. code-block:: python

   # Convert latent coordinates to numpy
   coords_vanilla = latent_coords.T.cpu().numpy()
   coords_pca = latent_coords_pca.T.cpu().numpy()

   # Create labels for the two clusters
   labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

   # Plot results
   fig, axes = plt.subplots(1, 2, figsize=(15, 6))

   # VanillaGTM results
   scatter1 = axes[0].scatter(coords_vanilla[0], coords_vanilla[1],
                             c=labels, cmap='viridis', alpha=0.6)
   axes[0].set_title('VanillaGTM (Random Initialization)')
   axes[0].set_xlabel('Latent Dimension 1')
   axes[0].set_ylabel('Latent Dimension 2')
   axes[0].grid(True, alpha=0.3)

   # GTM results
   scatter2 = axes[1].scatter(coords_pca[0], coords_pca[1],
                             c=labels, cmap='viridis', alpha=0.6)
   axes[1].set_title('GTM (PCA Initialization)')
   axes[1].set_xlabel('Latent Dimension 1')
   axes[1].set_ylabel('Latent Dimension 2')
   axes[1].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Understanding Model Parameters
-----------------------------

Key GTM Parameters
~~~~~~~~~~~~~~~~~

Let's explore how different parameters affect the model:

.. code-block:: python

   # Test different basis widths
   basis_widths = [0.1, 0.3, 0.5, 1.0]
   results = {}

   for width in basis_widths:
       model = GTM(
           num_nodes=100,
           num_basis_functions=25,
           basis_width=width,
           reg_coeff=0.01,
           max_iter=50,
           device="cuda" if torch.cuda.is_available() else "cpu"
       )
       model.fit(data)
       results[width] = model.training_history[-1]  # Final log-likelihood

   # Plot results
   plt.figure(figsize=(10, 6))
   widths, likelihoods = zip(*results.items())
   plt.plot(widths, likelihoods, 'bo-', linewidth=2, markersize=8)
   plt.xlabel('Basis Width')
   plt.ylabel('Final Log-Likelihood')
   plt.title('Effect of Basis Width on Model Performance')
   plt.grid(True, alpha=0.3)
   plt.show()

Grid Resolution Effects
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Test different grid resolutions
   node_counts = [25, 64, 100, 144]  # 5x5, 8x8, 10x10, 12x12
   bf_counts = [9, 16, 25, 36]       # 3x3, 4x4, 5x5, 6x6

   grid_results = {}

   for nodes, bfs in zip(node_counts, bf_counts):
       model = GTM(
           num_nodes=nodes,
           num_basis_functions=bfs,
           basis_width=0.3,
           reg_coeff=0.01,
           max_iter=50,
           device="cuda" if torch.cuda.is_available() else "cpu"
       )
       model.fit(data)
       grid_size = int(np.sqrt(nodes))
       grid_results[f"{grid_size}x{grid_size}"] = model.training_history[-1]

   print("Grid Resolution vs Performance:")
   for grid, likelihood in grid_results.items():
       print(f"{grid}: {likelihood:.2f}")

Working with New Data
--------------------

Once trained, you can use the model to project new data:

.. code-block:: python

   # Generate new test data
   new_data = torch.randn(100, n_features, dtype=torch.float64)

   # Project new data using the trained model
   new_responsibilities, new_log_likelihoods = gtm.project(new_data)
   new_latent_coords = gtm.transform(new_data)

   print(f"New data latent coordinates shape: {new_latent_coords.shape}")

   # Visualize new data with original data
   plt.figure(figsize=(10, 8))

   # Original data
   plt.scatter(coords_pca[0], coords_pca[1], c=labels, cmap='viridis',
               alpha=0.6, s=30, label='Training Data')

   # New data
   new_coords = new_latent_coords.T.cpu().numpy()
   plt.scatter(new_coords[0], new_coords[1], c='red', marker='x',
               s=50, label='New Data', alpha=0.8)

   plt.xlabel('Latent Dimension 1')
   plt.ylabel('Latent Dimension 2')
   plt.title('GTM Projection: Training vs New Data')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Model Persistence
-----------------

Save and load trained models:

.. code-block:: python

   # Save model state
   model_state = {
       'state_dict': gtm.state_dict(),
       'config': {
           'num_nodes': gtm.num_nodes,
           'num_basis_functions': gtm.num_basis_functions,
           'basis_width': gtm.basis_width,
           'reg_coeff': gtm.reg_coeff,
           'standardize': gtm.standardize
       },
       'standardizer_state': gtm.standardizer.__dict__ if gtm.standardizer else None
   }

   torch.save(model_state, 'gtm_model.pth')

   # Load model
   loaded_state = torch.load('gtm_model.pth')

   # Create new model with same configuration
   loaded_gtm = GTM(**loaded_state['config'])
   loaded_gtm.load_state_dict(loaded_state['state_dict'])

   # Restore standardizer if it was used
   if loaded_state['standardizer_state']:
       from chemographykit.gtm import DataStandardizer
       loaded_gtm.standardizer = DataStandardizer()
       loaded_gtm.standardizer.__dict__.update(loaded_state['standardizer_state'])

   print("Model loaded successfully!")

Best Practices
--------------

1. **Always use data standardization** for numerical stability
2. **Use GTM with PCA initialization** for better convergence
3. **Choose appropriate grid resolution** based on data complexity
4. **Monitor training convergence** using the training history
5. **Test different basis widths** to find optimal smoothness
6. **Use GPU acceleration** for large datasets

Common Issues and Solutions
--------------------------

**Slow Convergence**
   - Increase ``max_iter``
   - Adjust ``basis_width``
   - Use PCA initialization (GTM class)

**Memory Issues**
   - Reduce grid resolution (``num_nodes``)
   - Use CPU instead of GPU for very large datasets
   - Process data in batches

**Poor Mapping Quality**
   - Increase grid resolution
   - Adjust regularization (``reg_coeff``)
   - Try different basis widths
   - Ensure proper data standardization

Next Steps
----------

Now that you understand basic GTM training, you can:

1. Learn about :doc:`creating_landscapes` for visualization
2. Explore :doc:`classification_analysis` for supervised learning
3. Try :doc:`gpu_acceleration` for performance optimization

This tutorial covered the fundamentals of GTM training. In the next tutorials, you'll learn how to create informative visualizations and perform specific types of analysis with your trained models.
