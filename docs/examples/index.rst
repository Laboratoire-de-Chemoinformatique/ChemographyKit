Examples
========

This section provides practical examples demonstrating ChemographyKit's capabilities across different domains and use cases.

.. toctree::
   :maxdepth: 2
   :caption: Basic Examples

   simple_density_mapping
   binary_classification
   regression_analysis
   molecular_coordinates

.. toctree::
   :maxdepth: 2
   :caption: Chemical Informatics

   chemical_space_visualization
   compound_similarity
   scaffold_analysis
   property_prediction

.. toctree::
   :maxdepth: 2
   :caption: Advanced Applications

   dataset_comparison
   virtual_screening_workflow
   library_diversity_analysis
   multi_target_optimization

.. toctree::
   :maxdepth: 2
   :caption: Visualization Gallery

   interactive_landscapes
   publication_plots
   custom_styling
   comparative_analysis

Example Categories
==================

Basic Examples
--------------

These examples demonstrate fundamental ChemographyKit operations:

1. **Simple Density Mapping**: Basic density landscape creation and visualization
2. **Binary Classification**: Two-class classification landscape analysis
3. **Regression Analysis**: Continuous property mapping and visualization
4. **Molecular Coordinates**: Working with molecular coordinate calculations

Chemical Informatics
--------------------

Examples specific to chemical data analysis:

5. **Chemical Space Visualization**: Exploring chemical space with molecular descriptors
6. **Compound Similarity**: Using RP fingerprints for similarity analysis
7. **Scaffold Analysis**: Analyzing molecular scaffolds in GTM space
8. **Property Prediction**: Predicting molecular properties using GTM landscapes

Advanced Applications
--------------------

Complex workflows for real-world applications:

9. **Dataset Comparison**: Comparing chemical libraries and datasets
10. **Virtual Screening Workflow**: Complete virtual screening pipeline
11. **Library Diversity Analysis**: Analyzing and optimizing library diversity
12. **Multi-target Optimization**: Optimizing for multiple molecular properties

Visualization Gallery
--------------------

Comprehensive visualization examples:

13. **Interactive Landscapes**: Creating engaging interactive visualizations
14. **Publication Plots**: High-quality static plots for publications
15. **Custom Styling**: Advanced customization techniques
16. **Comparative Analysis**: Side-by-side comparison visualizations

Running the Examples
====================

Prerequisites
-------------

Make sure you have ChemographyKit installed with all dependencies:

.. code-block:: bash

   pdm install  # or pip install -e .

Some examples may require additional packages:

.. code-block:: bash

   # For chemical informatics examples
   pip install rdkit-pypi

   # For advanced visualization
   pip install seaborn

Data Files
----------

Example data files are provided in the repository:

- ``examples/data/synthetic/``: Synthetic datasets for testing
- ``examples/data/chemical/``: Chemical datasets (if available)
- ``examples/data/biological/``: Biological datasets (if available)

You can also generate synthetic data using the provided utilities.

Interactive Examples
===================

Many examples are available as Jupyter notebooks in the ``examples/notebooks/`` directory. These provide an interactive way to explore the code and experiment with parameters.

To run the notebooks:

.. code-block:: bash

   jupyter lab examples/notebooks/

Example Structure
================

Each example follows a consistent structure:

1. **Introduction**: Problem description and objectives
2. **Data Preparation**: Loading and preprocessing data
3. **Model Training**: GTM model configuration and training
4. **Analysis**: Specific analysis techniques
5. **Visualization**: Creating informative plots
6. **Interpretation**: Understanding and interpreting results
7. **Extensions**: Ideas for further exploration

Code Style
==========

All examples follow these conventions:

- Clear variable names and comments
- Modular code structure
- Error handling where appropriate
- Performance considerations noted
- Alternative approaches discussed

Contributing Examples
====================

We welcome contributions of new examples! Please:

1. Follow the established structure and style
2. Include clear documentation and comments
3. Test your code thoroughly
4. Provide sample data or generation code
5. Submit a pull request with your example

Getting Help
============

If you have questions about the examples:

1. Check the :doc:`../tutorials/index` for detailed explanations
2. Review the :doc:`../api/gtm` for API documentation
3. Submit issues on `GitHub <https://github.com/your-username/ChemographyKit/issues>`_
4. Contact the maintainers at varnek@unistra.fr

Performance Notes
================

Some examples may be computationally intensive. Consider:

- Using GPU acceleration when available
- Reducing dataset size for initial testing
- Adjusting grid resolution based on your hardware
- Using appropriate data types (float32 vs float64)

Reproducibility
==============

All examples include random seed settings for reproducible results. However, note that:

- GPU computations may have slight numerical differences
- Different PyTorch versions may produce slightly different results
- Hardware-specific optimizations may affect reproducibility

Next Steps
==========

After exploring the examples:

1. Try modifying parameters to see their effects
2. Apply the techniques to your own data
3. Combine multiple examples for complex workflows
4. Contribute your own examples to the community

The examples are designed to be educational and practical, providing a solid foundation for using ChemographyKit in your own projects.
