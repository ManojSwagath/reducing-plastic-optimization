You are a research engineer + applied mathematician helping me build a research-grade GitHub repository (not a product, not a web app).

Project Title

Plastic Bottle Shape Optimization using Surfaces of Revolution, Constrained Optimization, and Data-Driven Constraints

Core Goal

Minimize plastic material usage (surface area) of a water bottle while maximizing or maintaining volume, under physical and structural constraints.

Mathematical Foundation (MANDATORY)

Model the bottle as a surface of revolution:

Let 
𝑓
(
𝑥
)
f(x) be a parameterized curve (start with simple functions: polynomial / spline / sine-based).

Rotate 
𝑓
(
𝑥
)
f(x) about the vertical axis to form a 3D bottle.

Derive and implement:

Surface area formula of the surface of revolution

Volume formula of the surface of revolution

Define an optimization problem:

Objective: minimize surface area

Constraint: volume ≥ required volume

Variables: height and width parameters

Solve the constrained optimization using Lagrange multipliers.

Plot:

Bottle shapes

Surface area vs volume trade-offs

Machine Learning Component (CONSTRAINT LEARNING, NOT REPLACEMENT)

Assume I will provide real water bottle profiles (radius vs height data).

Use ML only to learn constraints, not to “find the optimal shape”.

Tasks:

Curve fitting on real bottle profiles

Identify common geometric function families

Infer minimum feasible surface-area thresholds required for structural strength

Use simple, interpretable ML:

Regression

Clustering

No deep learning unless absolutely necessary

Repository Style (VERY IMPORTANT)

This is a research repository, not an app

No frontend

No deployment

No unnecessary frameworks

Code clarity > performance

Required Repository Structure

Generate code and explanations that fit this structure:

plastic-bottle-shape-optimization/
├── README.md
├── theory/
│   ├── surface_of_revolution.md
│   ├── optimization_formulation.md
│   └── lagrange_multipliers.md
├── data/
│   ├── raw_profiles/
│   └── processed_profiles/
├── notebooks/
│   ├── 01_geometry_modeling.ipynb
│   ├── 02_surface_volume_analysis.ipynb
│   ├── 03_lagrange_optimization.ipynb
│   ├── 04_real_bottle_curve_fitting.ipynb
│   └── 05_constraint_learning_ml.ipynb
├── src/
│   ├── geometry.py
│   ├── optimization.py
│   ├── constraints.py
│   └── visualization.py
└── requirements.txt
README Requirements

Write the README like a mini research paper, including:

Problem Statement

Mathematical Modeling

Optimization Objective

Constraints (Analytical + Data-Driven)

Experiments & Observations

Limitations & Future Work

Tone & Style

Research-oriented

Precise

Honest about assumptions and limitations

No marketing language

Output Expectations

Generate clean Python code

Use NumPy, SciPy, Matplotlib, scikit-learn

Comment code where mathematical reasoning is involved

Assume I will run everything in VS Code

Begin by designing the mathematical model and folder structure, then proceed notebook by notebook.