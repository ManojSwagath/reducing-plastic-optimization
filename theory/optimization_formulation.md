# Optimization Formulation

## Problem Statement

**Goal**: Design a water bottle with minimum material usage (surface area) that meets volume requirements and physical constraints.

**Mathematical Formulation**:

$$
\begin{align}
\min_{\mathbf{p}} \quad & S(\mathbf{p}) \\
\text{subject to} \quad & V(\mathbf{p}) \geq V_{\text{required}} \\
& f(x; \mathbf{p}) \geq r_{\text{min}} \quad \forall x \in [0, H] \\
& r_{\text{min}} \leq r_i \leq r_{\text{max}} \quad \text{(parameter bounds)} \\
& H_{\text{min}} \leq H \leq H_{\text{max}} \\
& |f'(x; \mathbf{p})| \leq s_{\text{max}} \quad \text{(slope constraint)}
\end{align}
$$

where:
- $\mathbf{p} = [p_1, p_2, \ldots, p_n]$ are optimization parameters (e.g., spline control points)
- $S(\mathbf{p})$ is the surface area functional
- $V(\mathbf{p})$ is the volume functional
- $f(x; \mathbf{p})$ is the parametric bottle profile

## Variables

### Design Parameters $\mathbf{p}$

For **B-spline parametrization** with 6 control points:
$$\mathbf{p} = [r_0, r_1, r_2, r_3, r_4, r_5, H]$$

where:
- $r_i$ are radii at uniformly spaced heights
- $H$ is total bottle height (can be optimized or fixed)

**Typical ranges**:
- $r_i \in [1, 6]$ cm (base to neck radius)
- $H \in [10, 30]$ cm (practical bottle heights)

### Fixed Parameters
- $V_{\text{required}}$: Target volume (e.g., 250 mL, 500 mL, 1000 mL)
- $r_{\text{min}} = 0.5$ cm: Minimum radius (structural integrity)
- $s_{\text{max}} = 3$: Maximum slope (manufacturability)

## Objective Function

**Minimize surface area**:

$$S(\mathbf{p}) = 2\pi \int_0^H f(x; \mathbf{p}) \sqrt{1 + [f'(x; \mathbf{p})]^2} \, dx$$

### Why Minimize Surface Area?

1. **Material cost**: Directly proportional to surface area (wall thickness assumed uniform)
2. **Environmental impact**: Less plastic per bottle
3. **Manufacturing efficiency**: Fewer raw materials needed
4. **Weight**: Lighter bottles reduce shipping costs

### Alternative Objectives (Future Work)

- **Multi-objective**: Minimize $S + w_1 \cdot E_{\text{ergonomic}} - w_2 \cdot V$ (Pareto optimization)
- **Cost function**: Material cost + mold complexity + production time
- **Structural**: Maximize strength-to-weight ratio

## Constraints

### 1. Volume Constraint (Equality)

In practice, we want exactly the target volume:
$$V(\mathbf{p}) = V_{\text{required}}$$

Implemented as inequality for numerical stability:
$$V(\mathbf{p}) \geq V_{\text{required}}$$

The optimizer will naturally push toward equality (minimizing SA increases tight-fitting shapes).

### 2. Non-negativity Constraint

Ensure physical realizability:
$$f(x; \mathbf{p}) \geq r_{\text{min}} > 0 \quad \forall x \in [0, H]$$

Enforced by:
- **Parameter bounds**: $r_i \geq r_{\text{min}}$
- **Spline properties**: Cubic B-splines with positive control points remain positive

### 3. Slope Constraint (Manufacturability)

Limit steepness to ensure:
- Smooth blow-molding process
- No structural weak points
- Ergonomic grip

$$\left|\frac{df}{dx}\right| \leq s_{\text{max}} \approx 3$$

Steep slopes ($|f'| > 3$) create:
- Difficult mold extraction
- Material thinning during blow-molding
- Uncomfortable edges

### 4. Boundary Conditions

Optionally fix base and opening radii:
$$f(0) = r_{\text{base}}, \quad f(H) = r_{\text{top}}$$

Example:
- $r_{\text{base}} = 3$ cm (stable base)
- $r_{\text{top}} = 2$ cm (standard cap size)

## Connection to Classical Problems

### Isoperimetric Problem

Classical question: "Enclose maximum area with fixed perimeter."
- **2D answer**: Circle
- **3D answer**: Sphere

Our problem is the **dual**:
- "Enclose fixed volume with minimum surface area."
- **Unconstrained answer**: Sphere
- **With constraints**: Sub-optimal but practical shapes

### Dido's Problem

Legend: Queen Dido maximized land area using fixed-length rope from the sea.
- Solution: Semicircle
- Our bottle: Constrained by height limits, opening size, stability

### Calculus of Variations

General form: Find function $y(x)$ that extremizes
$$J[y] = \int_a^b F(x, y, y') \, dx$$

Our bottle:
- Functional: $S[f] = 2\pi \int_0^H f(x)\sqrt{1 + (f')^2} \, dx$
- Euler-Lagrange equation would give analytical optimality conditions
- **Practical approach**: Discretize with finite parameters, use numerical optimization

## Optimization Method: SLSQP

**Sequential Least Squares Programming** (SLSQP) from scipy.optimize:

### Why SLSQP?

1. **Handles constraints naturally**: Inequality and equality constraints
2. **Gradient-based**: Efficient for smooth objectives (our integrals are smooth)
3. **Proven method**: Widely used in engineering optimization
4. **Scipy integration**: Built-in, well-tested implementation

### Algorithm Overview

1. Linearize constraints around current point $\mathbf{p}_k$
2. Solve quadratic programming (QP) sub-problem:
   $$\min_{\Delta \mathbf{p}} \nabla S \cdot \Delta \mathbf{p} + \frac{1}{2} \Delta \mathbf{p}^T \mathbf{H} \Delta \mathbf{p}$$
   subject to linearized constraints
3. Update: $\mathbf{p}_{k+1} = \mathbf{p}_k + \alpha \Delta \mathbf{p}$ (line search for $\alpha$)
4. Repeat until convergence

### Alternatives

- **Global methods** (if local minima are problematic):
  - `scipy.optimize.differential_evolution` (genetic algorithm)
  - `scipy.optimize.basinhopping` (Monte Carlo + local optimization)
- **Gradient-free** (if derivative calculation is unstable):
  - `scipy.optimize.COBYLA` (Constrained Optimization BY Linear Approximations)

## Numerical Implementation Strategy

### Step 1: Parametrize Profile

```python
from scipy.interpolate import CubicSpline

def create_profile(params):
    """params = [r0, r1, ..., r5, H]"""
    radii = params[:-1]
    H = params[-1]
    heights = np.linspace(0, H, len(radii))
    return CubicSpline(heights, radii, bc_type='clamped')
```

### Step 2: Define Objective and Constraint Functions

```python
def objective(params):
    """Surface area to minimize"""
    profile = create_profile(params)
    return compute_surface_area(profile)

def volume_constraint(params):
    """Returns V(params) - V_required (should be >= 0)"""
    profile = create_profile(params)
    return compute_volume(profile) - V_target
```

### Step 3: Set Up Optimization

```python
from scipy.optimize import minimize, NonlinearConstraint

constraints = [
    {'type': 'ineq', 'fun': volume_constraint},  # V >= V_target
]

bounds = [(1, 6)] * 6 + [(10, 30)]  # radii and height bounds

result = minimize(
    objective,
    x0=initial_params,
    method='SLSQP',
    constraints=constraints,
    bounds=bounds,
    options={'maxiter': 200, 'ftol': 1e-6}
)
```

### Step 4: Extract Results

```python
optimal_params = result.x
final_surface_area = result.fun
lagrange_multiplier = result.mult  # KKT multipliers
converged = result.success
```

## Interpreting Results

### Lagrange Multiplier $\lambda$

From KKT conditions, $\lambda$ represents:
- **Marginal cost**: How much SA increases per unit volume
- **Shadow price**: If we relax volume by 1 mL, SA decreases by $\lambda$ cm²

**Example**: If $\lambda = 0.5$ cm²/mL, increasing target volume by 10 mL adds ~5 cm² SA.

### Optimality Conditions

At the solution $\mathbf{p}^*$:
1. $\nabla S(\mathbf{p}^*) = \lambda \nabla V(\mathbf{p}^*)$ (Lagrange condition)
2. $V(\mathbf{p}^*) = V_{\text{required}}$ (active constraint)
3. All bounds respected

### Sensitivity Analysis

How robust is the solution?
- Perturb constraints: vary $V_{\text{required}}$ by ±10%, re-optimize
- Plot trade-off curve: SA vs. V
- Identify which constraints are **active** (binding) vs. **inactive**

## Expected Results

### Theoretical Benchmarks

**Sphere** (optimal unconstrained):
$$S_{\text{sphere}} = \left(36\pi V^2\right)^{1/3}$$

For $V = 500$ mL:
$$S_{\text{sphere}} = (36\pi \cdot 500^2)^{1/3} \approx 238 \text{ cm}^2$$

**Cylinder** (baseline):
For $r = 3$ cm, $h = 17.7$ cm (to get $V = 500$ mL):
$$S_{\text{cylinder}} = 2\pi r h = 2\pi \cdot 3 \cdot 17.7 \approx 333 \text{ cm}^2$$

**Expected optimized bottle**:
- SA between sphere and cylinder: $250\text{-}280$ cm²
- Savings vs. cylinder: $15\text{-}25\%$
- Shape: Rounded base, tapered body, narrow neck

## Validation Checklist

- [ ] Optimizer converges (`result.success == True`)
- [ ] Volume constraint satisfied: $|V - V_{\text{target}}| < 1$ mL
- [ ] SA is lower than cylindrical baseline
- [ ] Profile is smooth (no kinks): $\max|f''(x)| < 10$ cm⁻¹
- [ ] Physically reasonable shape: gradual taper, stable base

## References

1. Nocedal & Wright, "Numerical Optimization" (2006), Chapter 12 (Theory of Constrained Optimization)
2. Kraft, "Algorithm 733: TOMP–Fortran modules for optimal control calculations" (1994) - SLSQP paper
3. Weisstein, Eric W. "Isoperimetric Problem." MathWorld
4. Mohammadi & Pironneau, "Applied Shape Optimization for Fluids" (2010)

---

**Next**: [Lagrange Multipliers](lagrange_multipliers.md)
