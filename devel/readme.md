# Development Guide

## Publishing to PyPI

This package uses GitHub Actions for automated publishing to PyPI. To publish a new version:

1. First, ensure you have the PyPI trusted publishing setup:
   - Go to PyPI project settings
   - Under "Publishing", select "Add a new pending publisher"
   - Set the publisher to this GitHub repository
   - Set the workflow name to publish.yml
   - Set the environment to "pypi"

2. To publish a new version:
   - Go to the Actions tab in GitHub
   - Select the "Publish to PyPI" workflow
   - Click "Run workflow"
   - Enter the new version number
   - Click "Run workflow"

The workflow will:
- Update version numbers in pyproject.toml and __init__.py
- Build wheels for multiple Python versions and platforms
- Upload the package to PyPI using trusted publishing
