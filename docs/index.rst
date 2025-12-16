ChemographyKit Documentation
====================

.. image:: https://img.shields.io/badge/License-MIT-brightgreen
   :alt: License
   :target: https://github.com/your-username/ChemographyKit/blob/main/LICENSE

.. image:: https://img.shields.io/badge/Python-3.11+-blue
   :alt: Python

.. image:: https://img.shields.io/badge/PyTorch-2.7.1%2B-red
   :alt: PyTorch

.. image:: https://img.shields.io/badge/GPU-CUDA_Ready-9cf
   :alt: GPU

**Mapping high-dimensional biology & chemistry into intuitive, navigable spaces with Generative Topographic Mapping (GTM).**

ChemographyKit is a comprehensive Python library for exploring chemical space and high-dimensional data using **Generative Topographic Mapping (GTM)**. GTM is a probabilistic dimensionality reduction technique that creates non-linear mappings from high-dimensional data spaces to interpretable low-dimensional latent spaces using a generative model with radial basis functions.

This is a **PyTorch-based** implementation of the GTM algorithm that runs on **GPU**, and includes functions for building landscapes and GTM-specific metrics.

.. note::
   Pair GTM maps with interactive notebooks or dashboards to let users zoom from global chemical space down to neighborhood-level structure–activity patterns.

Key Features
============

✅ **GPU-Accelerated**: PyTorch-based implementation with CUDA support for fast computation

🧠 **Multiple GTM Variants**:
   * ``VanillaGTM``: Basic GTM implementation with random initialization
   * ``GTM``: Enhanced version with PCA-based initialization for better convergence

📊 **Comprehensive Visualization**:
   * Interactive landscapes using Plotly (smooth heatmaps)
   * Static visualizations using Altair (discrete grid-based plots)
   * Support for density, classification, and regression landscapes

🔬 **Advanced Analytics**:
   * Responsibility Patterns (RP) for chemical space coverage analysis
   * Classification and regression landscape analysis

Quick Start
===========

Installation
------------

Using PDM (Recommended)::

   git clone <repository-url>
   cd ChemographyKit
   pdm install

Using pip::

   pip install numpy>=2.3.2 torch>=2.7.1 pandas>=2.3.2 altair>=5.5.0 plotly>=6.3.0 scikit-learn>=1.7.1 tqdm>=4.67.1

Basic Usage
-----------

.. code-block:: python

   import torch
   import numpy as np
   from chemographykit.gtm import GTM

   # Generate sample data
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
   responsibilities, log_likelihoods = gtm.project(data)

Documentation Contents
======================

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   quickstart
   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/gtm
   api/metrics
   api/utils
   api/plots

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog
   license

Applications of GTM
==================

GTM has been applied across biological data and extensively studied for analyzing large chemical datasets and exploring chemical space, including virtual screening, library comparison/design, de novo compound design, and multi-scale visualization.

Biological Datasets
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Domain
     - Representative Study
   * - **Genomes**
     - `Molecular Informatics (2023) <https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202300263>`_
   * - **Proteins**
     - `Bioinformatics (2022) <https://academic.oup.com/bioinformatics/article/38/8/2307/6528316>`_
   * - **Peptides**
     - `bioRxiv (2024) <https://www.biorxiv.org/content/10.1101/2024.11.17.622654v1.abstract>`_

Chemical Space & Big Chemical Data
----------------------------------

**Virtual screening**

* `Molecular Informatics (2018) <https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201800166>`_
* `European Journal of Medicinal Chemistry (2019) <https://linkinghub.elsevier.com/retrieve/pii/S0223-5234%2819%2930016-9>`_

**Library comparison & design**

* `Molecular Informatics (2011) <https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201100163>`_
* `Journal of Chemical Information and Modeling (2015) <https://pubs.acs.org/doi/10.1021/ci500575y>`_
* `PubMed (2019) <https://pubmed.ncbi.nlm.nih.gov/31407224/>`_
* `Molecular Informatics (2021) <https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202100289>`_
* `JCIM (2023) <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c00520>`_

**De novo design of chemical compounds**

* `JCIM (2019) <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00751>`_

**Multi-scale visualization of large chemical spaces**

* `JCIM (2022) <https://pubs.acs.org/doi/10.1021/acs.jcim.2c00509>`_

Citation
========

If you use ChemographyKit in your research, please cite our work::

   @software{chemographykit2025,
       title = {ChemographyKit: A Python Library for Generative Topographic Mapping},
       author = {Akhmetshin, Tagir and Plyer, Louis and Orlov, Alexey and Varnek, Alexandre},
       year = {2025},
       url = {https://github.com/your-username/ChemographyKit}
   }

Contact
=======

Contact: varnek@unistra.fr

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
