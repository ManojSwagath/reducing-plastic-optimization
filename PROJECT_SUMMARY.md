# Project Summary: Plastic Bottle Shape Optimization

## What This Project Is
This is a research repository focused on reducing plastic use in water bottles through mathematical shape optimization.

The bottle is modeled as a surface of revolution generated from a radius profile function along height.

## Core Goal
Minimize bottle surface area (proxy for plastic material usage) while meeting a required volume and practical geometric/manufacturing constraints.

## Mathematical Model
The bottle profile is defined as a function f(x), where:
- x is height
- f(x) is radius at height x

Key equations:
- Surface area: S = 2pi * integral[f(x) * sqrt(1 + (f'(x))^2)] dx
- Volume: V = pi * integral[f(x)^2] dx

## Optimization Strategy
- Objective: minimize S
- Main constraint: V >= target volume
- Additional constraints: radius bounds, height bounds, slope/smoothness bounds
- Solver: SciPy SLSQP (constrained nonlinear optimization)
- Theoretical interpretation: Lagrange multipliers and KKT conditions

## Data and Learning Component
The repository includes synthetic bottle profile data (height vs radius CSV files).

Machine learning is used only to learn feasible design constraints from data, such as:
- slope limits
- radius bounds by height region
- geometric ratio ranges
- empirical SA scaling with V^(2/3)

ML methods are interpretable (regression and clustering), not deep learning.

## Repository Workflow
1. Build and validate geometry calculations
2. Analyze surface area-volume tradeoffs
3. Run constrained optimization
4. Fit/analyze real or synthetic profile families
5. Learn and apply data-driven constraints

## Main Code Modules
- src/geometry.py: profile representation, SA/volume integration, validation
- src/optimization.py: constrained optimization and sensitivity utilities
- src/constraints.py: data-driven feasibility learning pipeline
- src/visualization.py: 2D/3D and tradeoff plotting

## Notebooks
- notebooks/01_geometry_modeling.ipynb
- notebooks/02_surface_volume_analysis.ipynb
- notebooks/03_lagrange_optimization.ipynb
- notebooks/04_real_bottle_curve_fitting.ipynb
- notebooks/05_constraint_learning_ml.ipynb

## Current Observations
- The project is well structured for a research workflow.
- Documentation is strong and theory-aligned.
- There are a few notebook-to-module API mismatches that should be patched before full notebook execution.

## Practical Outcome
The intended outcome is a reproducible research pipeline for proposing bottle shapes that use less plastic while remaining physically and manufacturing-feasible.
