# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Install dependencies:
  - Linux/macOS: `pip install -r requirements.txt`
  - Windows: `pip install -r requirements.txt`
- Create virtual environment:
  - Linux/macOS: `python -m venv venv && source venv/bin/activate`
  - Windows: `python -m venv venv && venv\Scripts\activate`
- Run the notebook:
  - Linux/macOS: `jupyter notebook src/aps1_fixed.ipynb`
  - Windows: `jupyter notebook src\aps1_fixed.ipynb`
- Generate predictions: Run all cells in the notebook sequentially

## Code Style Guidelines
- Imports: Group by standard library, third-party (pandas/numpy/sklearn), then local modules
- Formatting: Use descriptive variable/function names; document functions with docstrings
- Error handling: Use try/except blocks for data loading operations with specific exceptions
- Types: Follow scikit-learn typing conventions; use type hints where appropriate
- Naming: snake_case for functions/variables; PascalCase for classes
- Documentation: Add comments for complex data transformations and modeling decisions

## Dataset Requirements
- CSV output must match input format with only predictions column added
- Preserve all original rows and their order in output files
- Handle missing data appropriately in preprocessing steps