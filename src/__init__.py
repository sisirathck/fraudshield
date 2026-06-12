"""
src/__init__.py
===============
This file makes the src/ directory a Python "package".

WHAT IS A PYTHON PACKAGE?
    A package is just a folder that Python treats as importable.
    Without this file, you couldn't write:
        from src.data_loader import load_transactions

    With this file, that import works from anywhere in the project.

    It can be completely empty (like this) and still do its job.
    Some projects put shared constants here — we keep it empty for simplicity.

PACKAGE STRUCTURE:
    src/
    ├── __init__.py          ← This file
    ├── data_loader.py       ← Load data from the database
    ├── feature_engineering.py ← Create ML-ready features
    ├── model.py             ← ML model (from scratch + sklearn)
    └── visualizations.py   ← Reusable chart functions
"""
