Contributing to ChemographyKit
======================

We welcome contributions to ChemographyKit! This guide will help you get started with contributing to the project.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally::

    git clone https://github.com/your-username/ChemographyKit.git
    cd ChemographyKit

3. **Set up the development environment**::

    pdm install --dev

4. **Create a feature branch**::

    git checkout -b feature/your-feature-name

Types of Contributions
---------------------

We welcome several types of contributions:

Bug Reports
~~~~~~~~~~~

When reporting bugs, please include:

- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your system information (OS, Python version, ChemographyKit version)
- Minimal code example demonstrating the bug

Feature Requests
~~~~~~~~~~~~~~~

For new features:

- Describe the feature and its use case
- Explain why it would be valuable to the community
- Consider providing a rough implementation plan
- Discuss the feature in an issue before starting work

Documentation Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~

Documentation contributions are highly valued:

- Fix typos or unclear explanations
- Add missing docstrings
- Improve tutorials and examples
- Translate documentation

Code Contributions
~~~~~~~~~~~~~~~~~

Code contributions should:

- Follow the project's coding standards
- Include appropriate tests
- Update documentation as needed
- Pass all quality checks

Development Setup
----------------

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~

The development environment includes additional tools:

.. code-block:: bash

   pdm install --dev

This installs:

- **Testing**: pytest, pytest-cov
- **Code Quality**: black, isort, ruff, mypy, bandit
- **Pre-commit**: pre-commit hooks for automated checks
- **Documentation**: sphinx and related packages

Pre-commit Hooks
~~~~~~~~~~~~~~~

Set up pre-commit hooks to automatically check your code:

.. code-block:: bash

   pdm run pre-commit install

This will run quality checks before each commit.

Code Style
----------

We use several tools to maintain code quality:

Formatting
~~~~~~~~~~

- **Black**: Code formatting
- **isort**: Import sorting

Run formatting:

.. code-block:: bash

   pdm run black src/ tests/
   pdm run isort src/ tests/

Linting
~~~~~~~

- **Ruff**: Fast Python linter
- **MyPy**: Type checking
- **Bandit**: Security analysis

Run linting:

.. code-block:: bash

   pdm run ruff check src/ tests/
   pdm run mypy src/
   pdm run bandit -r src/

Quality Checks
~~~~~~~~~~~~~

Run all quality checks:

.. code-block:: bash

   pdm run pre-commit run --all-files

Testing
-------

We use pytest for testing. Tests are located in the ``tests/`` directory.

Running Tests
~~~~~~~~~~~~

Run the full test suite:

.. code-block:: bash

   pdm run pytest

Run with coverage:

.. code-block:: bash

   pdm run pytest --cov=src/chemographykit --cov-report=html

Run specific tests:

.. code-block:: bash

   pdm run pytest tests/test_gtm_core.py
   pdm run pytest -k "test_gtm_training"

Test Categories
~~~~~~~~~~~~~~

Tests are organized into categories using markers:

- ``@pytest.mark.slow``: Slow tests (skip with ``-m "not slow"``)
- ``@pytest.mark.gpu``: GPU-required tests
- ``@pytest.mark.integration``: Integration tests
- ``@pytest.mark.performance``: Performance benchmarks

Writing Tests
~~~~~~~~~~~~

When adding new features:

1. **Write tests first** (TDD approach recommended)
2. **Test both happy path and edge cases**
3. **Use appropriate fixtures** for setup
4. **Mock external dependencies** when needed
5. **Add performance tests** for critical functions

Example test structure:

.. code-block:: python

   import pytest
   import torch
   from chemographykit.gtm import GTM

   class TestGTMTraining:
       """Test GTM training functionality."""

       @pytest.fixture
       def sample_data(self):
           """Generate sample data for testing."""
           return torch.randn(100, 10, dtype=torch.float64)

       def test_gtm_initialization(self):
           """Test GTM model initialization."""
           gtm = GTM(num_nodes=25, num_basis_functions=9)
           assert gtm.num_nodes == 25
           assert gtm.num_basis_functions == 9

       def test_gtm_training(self, sample_data):
           """Test GTM training process."""
           gtm = GTM(num_nodes=25, num_basis_functions=9, max_iter=10)
           gtm.fit(sample_data)
           assert len(gtm.training_history) <= 10

Documentation
-------------

We use Sphinx for documentation with the following extensions:

- **autodoc**: Automatic API documentation
- **napoleon**: Google/NumPy style docstrings
- **nbsphinx**: Jupyter notebook integration
- **myst_parser**: Markdown support

Building Documentation
~~~~~~~~~~~~~~~~~~~~~

Build documentation locally:

.. code-block:: bash

   cd docs/
   sphinx-build -b html . _build/html

Or use the make command:

.. code-block:: bash

   cd docs/
   make html

Docstring Style
~~~~~~~~~~~~~~

We follow the NumPy docstring convention:

.. code-block:: python

   def example_function(param1: int, param2: str = "default") -> bool:
       """
       Brief description of the function.

       Longer description if needed, explaining the purpose,
       algorithm, or important details.

       Parameters
       ----------
       param1 : int
           Description of param1.
       param2 : str, optional
           Description of param2 (default is "default").

       Returns
       -------
       bool
           Description of return value.

       Raises
       ------
       ValueError
           When param1 is negative.

       Examples
       --------
       >>> result = example_function(5, "test")
       >>> print(result)
       True

       Notes
       -----
       Additional notes about the function, algorithm details,
       or references to papers.
       """
       if param1 < 0:
           raise ValueError("param1 must be non-negative")
       return param1 > 0

Pull Request Process
-------------------

1. **Create a descriptive PR title** following conventional commits:

   - ``feat: add new GTM visualization method``
   - ``fix: resolve memory leak in large dataset processing``
   - ``docs: improve API documentation for metrics module``
   - ``test: add comprehensive tests for regression utils``

2. **Write a clear PR description**:

   - Explain what changes you made and why
   - Reference any related issues
   - Include screenshots for UI changes
   - Note any breaking changes

3. **Ensure all checks pass**:

   - All tests pass
   - Code coverage is maintained
   - Linting and formatting checks pass
   - Documentation builds successfully

4. **Request review** from maintainers

5. **Address feedback** promptly and courteously

6. **Squash commits** if requested before merging

Release Process
--------------

Releases follow semantic versioning (SemVer):

- **Major** (x.0.0): Breaking changes
- **Minor** (0.x.0): New features, backward compatible
- **Patch** (0.0.x): Bug fixes, backward compatible

Release steps:

1. Update version in ``pyproject.toml``
2. Update ``CHANGELOG.md``
3. Create release PR
4. Tag release after merge
5. GitHub Actions handles PyPI publishing

Community Guidelines
-------------------

Code of Conduct
~~~~~~~~~~~~~~

We follow the Python Community Code of Conduct. Be respectful, inclusive, and constructive in all interactions.

Communication
~~~~~~~~~~~~

- **GitHub Issues**: Bug reports, feature requests, questions
- **GitHub Discussions**: General discussion, ideas, help
- **Email**: Direct contact with maintainers (varnek@unistra.fr)

Recognition
~~~~~~~~~~

Contributors are recognized in:

- ``README.md`` contributors section
- Release notes for significant contributions
- Academic papers when appropriate

Getting Help
-----------

If you need help with contributing:

1. **Read this guide** thoroughly
2. **Check existing issues** and discussions
3. **Ask questions** in GitHub Discussions
4. **Contact maintainers** directly if needed

Common Issues
~~~~~~~~~~~~

**Tests failing locally but passing in CI**
   - Check Python version compatibility
   - Ensure all dependencies are installed
   - Clear pytest cache: ``pytest --cache-clear``

**Linting errors**
   - Run ``pdm run pre-commit run --all-files``
   - Check specific tool documentation
   - Ask for help if errors are unclear

**Documentation build failures**
   - Check Sphinx syntax in RST files
   - Ensure all imports work correctly
   - Verify notebook execution if using nbsphinx

Thank You!
---------

Thank you for contributing to ChemographyKit! Your contributions help make the project better for everyone in the scientific computing and cheminformatics communities.

Every contribution, no matter how small, is valuable and appreciated. Whether you're fixing a typo, adding a feature, or improving documentation, you're helping advance open-source scientific software.

Happy coding! 🚀
