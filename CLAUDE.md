# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Run Commands
- Install dependencies: `pip install -r requirements.txt`
- Run the notebook: `jupyter notebook src/aps1_final.ipynb`
- Generate predictions: Run all cells in the notebook sequentially

## Code Style Guidelines
- Imports: Group by standard library, third-party, and local modules
- Formatting: Use descriptive variable/function names; document functions with docstrings
- Error handling: Use try/except blocks for data loading operations
- Types: Follow scikit-learn typing conventions
- Naming: snake_case for functions/variables; PascalCase for classes
- Documentation: Add comments for complex data transformations

## Dataset Requirements
- CSV output must match input format with only predictions column added
- Preserve all original rows and their order in output files