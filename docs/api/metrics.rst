Metrics Module
==============

The metrics module provides specialized metrics for GTM analysis, including Responsibility Pattern (RP) fingerprints and coverage calculations.

.. automodule:: chemographykit.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

Responsibility Pattern Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: chemographykit.metrics.resp_to_pattern

.. autofunction:: chemographykit.metrics.compute_rp_coverage

.. autofunction:: chemographykit.metrics.get_fingerprint_counts

Usage Examples
--------------

Creating RP Fingerprints
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.metrics import resp_to_pattern
   import numpy as np

   # Example responsibility vector (from GTM projection)
   responsibilities = np.array([0.1, 0.05, 0.3, 0.2, 0.15, 0.1, 0.05, 0.05])

   # Convert to RP fingerprint
   rp_fingerprint = resp_to_pattern(
       responsibilities,
       n_bins=5,
       threshold=0.01
   )
   print(f"RP fingerprint: {rp_fingerprint}")

Computing Coverage Between Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.metrics import compute_rp_coverage
   import numpy as np

   # Generate example RP fingerprints for reference and test sets
   reference_fps = np.random.randint(0, 6, size=(1000, 100))  # 1000 molecules, 100 nodes
   test_fps = np.random.randint(0, 6, size=(500, 100))        # 500 molecules, 100 nodes

   # Compute weighted coverage
   weighted_coverage = compute_rp_coverage(
       reference_fps,
       test_fps,
       use_weight=True
   )
   print(f"Weighted coverage: {weighted_coverage:.3f}")

   # Compute unweighted coverage
   unweighted_coverage = compute_rp_coverage(
       reference_fps,
       test_fps,
       use_weight=False
   )
   print(f"Unweighted coverage: {unweighted_coverage:.3f}")

Working with Fingerprint Counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.metrics import get_fingerprint_counts
   import numpy as np

   # Example RP fingerprints
   fingerprints = np.array([
       [1, 2, 0, 3, 1],
       [1, 2, 0, 3, 1],  # Duplicate
       [2, 1, 3, 0, 2],
       [1, 2, 0, 3, 1],  # Another duplicate
   ])

   # Get counts of unique patterns
   pattern_counts = get_fingerprint_counts(fingerprints)
   print("Pattern counts:")
   for pattern, count in pattern_counts.items():
       print(f"  {pattern}: {count}")

Practical Workflow
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from chemographykit.gtm import GTM
   from chemographykit.metrics import resp_to_pattern, compute_rp_coverage
   import torch
   import numpy as np

   # Train GTM model
   data = torch.randn(1000, 50, dtype=torch.float64)
   gtm = GTM(num_nodes=100, num_basis_functions=25)
   gtm.fit(data)

   # Project data to get responsibilities
   responsibilities, _ = gtm.project(data)
   responsibilities_np = responsibilities.T.cpu().numpy()

   # Convert to RP fingerprints
   rp_fingerprints = np.array([
       resp_to_pattern(resp, n_bins=10, threshold=0.01)
       for resp in responsibilities_np
   ])

   # Split into reference and test sets
   n_ref = 700
   reference_fps = rp_fingerprints[:n_ref]
   test_fps = rp_fingerprints[n_ref:]

   # Compute coverage
   coverage = compute_rp_coverage(reference_fps, test_fps, use_weight=True)
   print(f"Dataset coverage: {coverage:.3f}")

Notes
-----

Responsibility Pattern (RP) Fingerprints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RP fingerprints are a unique feature of GTM that encode how molecules are distributed across the latent space. They provide:

- **Interpretable representation**: Each element corresponds to a specific region in chemical space
- **Coverage analysis**: Quantify how well one dataset covers another in chemical space
- **Similarity metrics**: Compare molecules based on their chemical space neighborhoods

The fingerprints are created by:

1. Binning responsibility values into discrete levels
2. Applying a threshold to focus on significant responsibilities
3. Creating a compact representation of the molecule's position in GTM space

Coverage Metrics
~~~~~~~~~~~~~~~~

Coverage metrics help answer questions like:

- How well does my training set cover my test set?
- What fraction of chemical space is represented in my library?
- How diverse is my compound collection?

The weighted coverage gives more importance to highly populated regions, while unweighted coverage treats all occupied regions equally.

Available Functions
~~~~~~~~~~~~~~~~~~

The metrics module provides these core functions:

- **resp_to_pattern**: Convert responsibility vectors to RP fingerprints
- **compute_rp_coverage**: Calculate coverage between two datasets
- **get_fingerprint_counts**: Count unique patterns in fingerprint arrays
