Installation
============

Requirements
------------

* Python ≥ 3.11
* PyTorch ≥ 2.7.1
* NumPy ≥ 2.3.2
* Pandas ≥ 2.3.2
* Scikit-learn ≥ 1.7.1
* Altair ≥ 5.5.0
* Plotly ≥ 6.3.0
* tqdm ≥ 4.67.1

Installation Methods
-------------------

Using PDM (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

PDM is the recommended package manager for ChemographyKit development and usage::

   git clone <repository-url>
   cd ChemographyKit
   pdm install

This will install ChemographyKit along with all required dependencies in a virtual environment.

Using pip
~~~~~~~~~

You can install ChemographyKit using pip. First, install the required dependencies::

   pip install numpy>=2.3.2 torch>=2.7.1 pandas>=2.3.2 altair>=5.5.0 plotly>=6.3.0 scikit-learn>=1.7.1 tqdm>=4.67.1

Then install ChemographyKit from source::

   git clone <repository-url>
   cd ChemographyKit
   pip install -e .

GPU Support
-----------

For GPU acceleration, ensure you have a CUDA-enabled build of PyTorch installed. Visit the `PyTorch installation guide <https://pytorch.org/get-started/locally/>`_ to select the appropriate version for your system.

To verify GPU support::

   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

Development Installation
-----------------------

For development purposes, install with additional development dependencies::

   git clone <repository-url>
   cd ChemographyKit
   pdm install --dev

This includes testing, linting, and documentation tools:

* pytest for testing
* black, isort, ruff for code formatting and linting
* mypy for type checking
* pre-commit for git hooks
* sphinx for documentation

Verification
-----------

To verify your installation, run the test suite::

   pdm run pytest tests/ -v

Or import ChemographyKit in Python::

   python -c "import chemographykit; print('ChemographyKit installed successfully!')"

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'torch'**
   Make sure PyTorch is properly installed. See the PyTorch installation guide.

**CUDA out of memory**
   Reduce batch size or use CPU by setting ``device="cpu"`` in GTM initialization.

**ModuleNotFoundError: No module named 'chemographykit'**
   Ensure you've installed ChemographyKit in your current Python environment.

Getting Help
~~~~~~~~~~~~

If you encounter installation issues:

1. Check the `GitHub Issues <https://github.com/your-username/ChemographyKit/issues>`_ page
2. Create a new issue with your system information and error message
3. Contact the maintainers at varnek@unistra.fr
