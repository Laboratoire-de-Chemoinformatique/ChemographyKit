# ChemographyKit Documentation

This document provides an overview of the comprehensive documentation system created for ChemographyKit, following the structure and best practices of the SynPlanner project.

## Documentation Structure

The documentation is built using **Sphinx** with the **Read the Docs theme** and includes:

### 📚 Core Documentation Files

```
docs/
├── conf.py                 # Sphinx configuration
├── index.rst              # Main documentation index
├── installation.rst       # Installation guide
├── quickstart.rst          # Quick start tutorial
├── contributing.rst        # Contribution guidelines
├── changelog.rst           # Version history
├── license.rst            # License information
└── requirements.txt       # Documentation dependencies
```

### 🔧 API Reference

```
docs/api/
├── gtm.rst                # Core GTM module documentation
├── metrics.rst            # Metrics and RP fingerprints
├── utils.rst              # Utility modules (classification, regression, density, molecules)
└── plots.rst              # Visualization modules (Plotly and Altair)
```

### 📖 Tutorials

```
docs/tutorials/
├── index.rst                    # Tutorial overview
├── basic_gtm_training.rst       # Comprehensive GTM training tutorial
└── [planned additional tutorials]
```

### 💡 Examples

```
docs/examples/
├── index.rst                    # Examples gallery overview
└── [planned example files]
```

### 🎨 Static Assets

```
docs/_static/
├── custom.css              # Custom CSS styling
└── [images and other assets]
```

### 🛠️ Build System

```
docs/
├── Makefile                # Unix build commands
├── make.bat                # Windows build commands
└── build.py                # Python build script with advanced features
```

## Key Features

### ✅ Professional Documentation Structure
- **Sphinx-based**: Industry-standard documentation system
- **Read the Docs theme**: Professional, responsive design
- **Auto-generated API docs**: Using autodoc for up-to-date API reference
- **Cross-references**: Automatic linking between sections
- **Search functionality**: Built-in documentation search

### 📊 Rich Content Support
- **Code examples**: Syntax-highlighted code blocks
- **Mathematical notation**: LaTeX math support via MathJax
- **Interactive elements**: Support for Jupyter notebooks
- **Multiple formats**: HTML, PDF, and ePub output

### 🔄 Automated Building
- **GitHub Actions**: Automated documentation building and deployment
- **Read the Docs**: Integration for hosted documentation
- **Live reload**: Development server with auto-refresh
- **Link checking**: Automated broken link detection

### 🎯 User-Focused Design
- **Progressive complexity**: From quick start to advanced topics
- **Real-world examples**: Practical use cases and applications
- **Comprehensive tutorials**: Step-by-step guides
- **API reference**: Complete function and class documentation

## Building the Documentation

### Local Development

1. **Install dependencies**:
   ```bash
   pdm install --dev
   ```

2. **Build HTML documentation**:
   ```bash
   cd docs/
   python build.py build
   ```

3. **Serve locally**:
   ```bash
   python build.py serve
   ```

4. **Live reload during development**:
   ```bash
   python build.py livehtml
   ```

### Advanced Build Options

- **Clean build**: `python build.py clean`
- **Check links**: `python build.py linkcheck`
- **PDF output**: `python build.py build --builder pdf`
- **Custom port**: `python build.py serve --port 8080`

## Deployment Options

### 1. Read the Docs (Recommended)
- Configuration: `.readthedocs.yaml`
- Automatic builds from Git commits
- Multiple format support (HTML, PDF, ePub)
- Version management
- Custom domain support

### 2. GitHub Pages
- GitHub Actions workflow: `.github/workflows/docs.yml`
- Automatic deployment on push to main branch
- Free hosting for public repositories

### 3. Self-Hosted
- Build locally or in CI/CD
- Deploy to any web server
- Full control over hosting environment

## Documentation Standards

### Writing Style
- **Clear and concise**: Easy to understand explanations
- **Code examples**: Working code snippets for all features
- **Progressive disclosure**: Basic to advanced information flow
- **Consistent formatting**: Standardized structure across sections

### API Documentation
- **NumPy docstring style**: Consistent parameter and return documentation
- **Type hints**: Full type information for all functions
- **Examples**: Usage examples for all public functions
- **Cross-references**: Links between related functions and classes

### Tutorial Structure
1. **Overview**: What the tutorial covers
2. **Prerequisites**: Required knowledge and setup
3. **Step-by-step instructions**: Clear, actionable steps
4. **Code examples**: Complete, runnable examples
5. **Visualization**: Plots and outputs where relevant
6. **Best practices**: Recommendations and tips
7. **Next steps**: Links to related tutorials

## Maintenance

### Regular Updates
- **API changes**: Update documentation when code changes
- **New features**: Add tutorials and examples for new functionality
- **Bug fixes**: Update examples that may be affected
- **Dependencies**: Keep documentation dependencies current

### Quality Assurance
- **Link checking**: Regular verification of external links
- **Example testing**: Ensure all code examples work
- **Spelling and grammar**: Regular proofreading
- **User feedback**: Incorporate feedback from users

## Extending the Documentation

### Adding New Tutorials
1. Create new `.rst` file in `docs/tutorials/`
2. Follow the established structure and style
3. Add to `docs/tutorials/index.rst` table of contents
4. Include working code examples
5. Test the tutorial thoroughly

### Adding New Examples
1. Create example in `docs/examples/`
2. Include complete, self-contained code
3. Provide clear explanations
4. Add to examples index
5. Consider creating Jupyter notebook version

### Customizing Appearance
- Modify `docs/_static/custom.css` for styling changes
- Update `docs/conf.py` for configuration changes
- Add images to `docs/_static/` directory
- Customize theme options in `conf.py`

## Integration with SynPlanner Patterns

The documentation follows SynPlanner's successful patterns:

### ✅ Comprehensive Coverage
- Complete API reference with examples
- Progressive tutorials from basic to advanced
- Real-world application examples
- Clear installation and setup instructions

### ✅ Professional Presentation
- Clean, modern design with Read the Docs theme
- Consistent formatting and structure
- Professional badge integration
- Clear navigation and organization

### ✅ Developer-Friendly
- Contributing guidelines and development setup
- Code quality standards and tools
- Automated testing and deployment
- Version control integration

### ✅ Community-Focused
- Clear communication channels
- Issue templates and support information
- Academic citation information
- License and attribution details

## Future Enhancements

### Planned Additions
- **Interactive tutorials**: Jupyter notebook integration
- **Video content**: Screencasts for complex workflows
- **API changelog**: Detailed API change tracking
- **Performance benchmarks**: Documented performance characteristics
- **Troubleshooting guide**: Common issues and solutions

### Community Contributions
- **Example gallery**: User-contributed examples
- **Use case studies**: Real-world application stories
- **Translation support**: Multi-language documentation
- **Plugin ecosystem**: Documentation for extensions

This comprehensive documentation system provides a solid foundation for ChemographyKit users and contributors, following the proven patterns established by successful scientific software projects like SynPlanner.
