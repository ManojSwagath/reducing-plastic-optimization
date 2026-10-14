# First Paper Draft

## Title
Geometric Modeling of Axisymmetric Plastic Bottles via Surfaces of Revolution: Validation, Parametric Exploration, and Sensitivity

## Abstract
This paper presents the first stage of a research pipeline for plastic bottle shape optimization. Bottle geometries are modeled as axisymmetric surfaces of revolution generated from radius profiles over height. We validate numerical surface-area and volume computations against analytical cylinder and cone references, explore polynomial and B-spline profile families, and quantify sensitivity to mid-body radius. Results confirm numerical correctness at validation scale and show measurable surface-area and volume responses to shape parameters, establishing a reliable computational base for subsequent constrained optimization and learned manufacturing constraints.

## 1. Problem Context
Reducing plastic usage in bottle production can be framed as a geometry problem: for a required volume, identify shapes with lower surface area while preserving practical form. This first paper focuses on mathematical modeling and numerical validation before full constrained optimization.

## 2. Mathematical Formulation
Let x denote height and f(x) denote bottle radius at x. Rotating f(x) around the vertical axis yields a surface of revolution.

Surface area (lateral):

$$
S = 2\pi \int_0^H f(x)\sqrt{1 + (f'(x))^2}\,dx
$$

Volume:

$$
V = \pi \int_0^H [f(x)]^2\,dx
$$

These functionals are implemented numerically in the geometry module.

## 3. Methods
- Profiles tested: cylinder, conical taper, polynomial profile, and multiple cubic B-spline profiles.
- Numerical integration: adaptive quadrature.
- Validation strategy: compare numerical outputs against analytical formulas for cylinder and truncated cone.
- Sensitivity experiment: vary mid-body radius and observe induced changes in S and V.

## 4. Results (Notebook 01)
### 4.1 Numerical Validation
Cylinder and cone validation runs showed near-zero relative error at notebook precision for both surface area and volume in tested configurations.

### 4.2 Baseline Profile Comparison
Representative baseline outputs:
- Cylinder (r=3.0, h=15.0): S=282.74 cm^2, V=424.12 mL, S/V=0.6667
- Cone (r0=4.0, r1=2.0, h=15.0): S=285.25 cm^2, V=439.82 mL, S/V=0.6485
- Polynomial profile: S=331.37 cm^2, V=580.80 mL, S/V=0.5705

Interpretation: tapered and smoother distributed profiles can improve S/V compared with a same-scale simple cylinder baseline.

### 4.3 B-Spline Family Exploration
Three B-spline profile families (standard, ergonomic, sports) were validated as physically admissible in the notebook and used for geometric comparison plots and table-based metric inspection.

### 4.4 Sensitivity to Mid-Body Radius
Observed notebook sensitivity summary:
- Mid-body radius perturbation: Delta r = 2.0 cm
- Surface area response: Delta S = 50.7 cm^2 (15.4 percent)
- Volume response: Delta V = 159.5 mL (34.2 percent)

This indicates strong coupling between central radius control and both material and capacity metrics.

### 4.5 Gap to Sphere Baseline (for tested B-spline examples)
Notebook-reported excess relative to theoretical sphere bounds:
- Standard bottle: +8.6 percent
- Ergonomic bottle: +10.6 percent
- Sports bottle: +6.6 percent

These values provide a geometric efficiency reference prior to constrained optimization.

## 5. Discussion
The stage-1 modeling pipeline is numerically stable for analytic references and expressive enough to represent diverse bottle morphologies. The sensitivity analysis shows that a small set of control radii can materially affect both objective and constraint quantities, supporting spline-based parameterization as a suitable choice for upcoming optimization stages.

## 6. Limitations
- Axisymmetric assumption excludes non-axisymmetric ergonomic features.
- Only geometric functionals are considered here; structural mechanics and manufacturing process constraints are not yet encoded in this stage.
- Reported comparisons are scenario-specific and serve as computational baselines, not final design prescriptions.

## 7. Next Step
Proceed to constrained optimization over parameterized profiles with explicit volume constraints, then integrate data-driven manufacturability bounds.

## Reproducibility Anchor
Primary execution source for this paper is Notebook 01.

- Notebook: notebooks/01_geometry_modeling.ipynb
- Geometry core: src/geometry.py
- Visualization utilities: src/visualization.py
