Changelog
=========

All notable changes to ChemographyKit will be documented in this file.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
------------

Added
~~~~~
- Comprehensive Sphinx documentation with API reference
- Interactive tutorials for all major features
- Read the Docs integration
- Example gallery with real-world applications

[0.1.0] - 2025-01-XX
--------------------

Added
~~~~~
- Initial release of ChemographyKit
- Core GTM implementations (VanillaGTM and GTM with PCA initialization)
- GPU acceleration with PyTorch backend
- Comprehensive visualization suite:

  - Interactive Plotly landscapes (smooth heatmaps)
  - Static Altair visualizations (discrete grids)
  - Support for density, classification, and regression landscapes

- Advanced analytics:

  - Responsibility Pattern (RP) fingerprints
  - Coverage and similarity metrics
  - Dataset comparison tools

- Utility modules:

  - Classification analysis tools
  - Regression analysis tools
  - Molecular coordinate calculations
  - Density matrix operations

- Robust testing suite with pytest
- Type hints throughout the codebase
- Development tools (black, ruff, mypy, pre-commit)
- Comprehensive API documentation

Features
~~~~~~~~

**Core GTM Algorithm**
- Probabilistic dimensionality reduction
- EM algorithm optimization
- GPU acceleration with CUDA support
- Data standardization with NaN handling
- PCA-based initialization for improved convergence

**Visualization Capabilities**
- Interactive landscapes with zoom and hover
- Publication-ready static plots
- Customizable color schemes and styling
- Molecular coordinate overlays
- Comparative analysis plots

**Analysis Tools**
- Binary and multi-class classification landscapes
- Continuous property regression analysis
- Responsibility pattern fingerprints for similarity
- Coverage metrics for dataset comparison
- Chemical space exploration utilities

**Performance Features**
- GPU acceleration for large datasets
- Memory-efficient processing
- Configurable precision (float32/float64)
- Batch processing capabilities
- Optimized grid operations

**Developer Experience**
- Comprehensive type hints
- Detailed docstrings with examples
- Modular architecture for extensibility
- Consistent API design
- Extensive testing coverage

Dependencies
~~~~~~~~~~~~
- Python ≥ 3.11
- PyTorch ≥ 2.7.1
- NumPy ≥ 2.3.2
- Pandas ≥ 2.3.2
- Scikit-learn ≥ 1.7.1
- Altair ≥ 5.5.0
- Plotly ≥ 6.3.0
- tqdm ≥ 4.67.1

Known Issues
~~~~~~~~~~~~
- GPU memory usage can be high for very large datasets
- Some numerical differences between CPU and GPU computations
- Plotly interactive plots may be slow with very dense grids

Future Releases
===============

Planned Features
---------------

**Version 0.2.0**
- Command-line interface for common workflows
- Additional GTM variants (hierarchical, supervised)
- Enhanced molecular descriptor support
- Batch processing utilities for very large datasets

**Version 0.3.0**
- Integration with popular cheminformatics libraries
- Advanced clustering and outlier detection
- Time-series GTM for temporal data
- Enhanced GPU memory management

**Version 1.0.0**
- Stable API with backward compatibility guarantees
- Comprehensive benchmarking suite
- Production-ready deployment tools
- Extended documentation with more use cases

Contributing
============

We welcome contributions! See our :doc:`contributing` guide for details on:

- Setting up the development environment
- Code style and quality standards
- Testing requirements
- Documentation standards
- Pull request process

License
=======

ChemographyKit is released under the MIT License. See the LICENSE file for details.

Acknowledgments
==============

ChemographyKit builds upon decades of research in:

- Generative Topographic Mapping (Bishop et al., 1998)
- Probabilistic dimensionality reduction techniques
- Chemical space analysis and visualization
- GPU-accelerated scientific computing

We thank the scientific community for their foundational work and the open-source community for the excellent tools that make ChemographyKit possible.
