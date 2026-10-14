# Lagrange Multipliers and Constrained Optimization

## Overview

**Lagrange multipliers** provide a method for finding extrema (minimum or maximum) of a function subject to constraints. For our bottle optimization, we minimize surface area while ensuring the volume meets requirements.

## The Method of Lagrange Multipliers

### Unconstrained Problem

Simple optimization:
$$\min_{\mathbf{x}} f(\mathbf{x})$$

Solution: Find where gradient vanishes:
$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

### Equality-Constrained Problem

Optimization with constraint:
$$
\begin{align}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{subject to} \quad & g(\mathbf{x}) = 0
\end{align}
$$

**Key insight**: At the optimum, the gradients of $f$ and $g$ must be parallel:
$$\nabla f(\mathbf{x}^*) = \lambda \nabla g(\mathbf{x}^*)$$

where $\lambda$ is the **Lagrange multiplier**.

### Why Are Gradients Parallel?

**Geometric intuition**:
- The constraint $g(\mathbf{x}) = 0$ defines a surface (manifold) in parameter space
- At any point on this surface, $\nabla g$ points perpendicular to the surface
- To minimize $f$ while staying on the constraint surface, we move tangent to the surface
- At the minimum, we cannot move tangent to the surface without increasing $f$
- This happens exactly when $\nabla f$ is also perpendicular to the surface
- Therefore, $\nabla f$ and $\nabla g$ are parallel: $\nabla f = \lambda \nabla g$

## The Lagrangian Function

Combine objective and constraint into a single function:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) - \lambda g(\mathbf{x})$$

**Optimality conditions** (necessary conditions for $\mathbf{x}^*$ to be a minimum):
1. $\nabla_{\mathbf{x}} \mathcal{L} = \nabla f - \lambda \nabla g = \mathbf{0}$ (stationary w.r.t. $\mathbf{x}$)
2. $\nabla_{\lambda} \mathcal{L} = -g(\mathbf{x}) = 0$ (constraint satisfaction)

These are $n + 1$ equations for $n$ variables plus $\lambda$.

## Application to Bottle Optimization

### Problem Setup

$$
\begin{align}
\min_{\mathbf{p}} \quad & S(\mathbf{p}) \\
\text{s.t.} \quad & V(\mathbf{p}) - V_{\text{req}} = 0
\end{align}
$$

Rewrite constraint as $g(\mathbf{p}) = V(\mathbf{p}) - V_{\text{req}} = 0$.

### Lagrangian

$$\mathcal{L}(\mathbf{p}, \lambda) = S(\mathbf{p}) - \lambda \left(V(\mathbf{p}) - V_{\text{req}}\right)$$

### Optimality Conditions

$$
\begin{cases}
\frac{\partial \mathcal{L}}{\partial p_i} = \frac{\partial S}{\partial p_i} - \lambda \frac{\partial V}{\partial p_i} = 0 & \text{for } i = 1, \ldots, n \\
\frac{\partial \mathcal{L}}{\partial \lambda} = V(\mathbf{p}) - V_{\text{req}} = 0
\end{cases}
$$

**Interpretation of first equation**:
$$\frac{\partial S}{\partial p_i} = \lambda \frac{\partial V}{\partial p_i}$$

At the optimum, the rate of change of surface area w.r.t. parameter $p_i$ is proportional to the rate of change of volume. The proportionality constant is $\lambda$.

## Inequality Constraints: KKT Conditions

Real problems have inequality constraints (e.g., $V(\mathbf{p}) \geq V_{\text{req}}$, $r_i \geq 0$).

### Karush-Kuhn-Tucker (KKT) Conditions

For problem:
$$
\begin{align}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g_i(\mathbf{x}) \geq 0, \quad i = 1, \ldots, m
\end{align}
$$

**Lagrangian**:
$$\mathcal{L}(\mathbf{x}, \boldsymbol{\lambda}) = f(\mathbf{x}) - \sum_{i=1}^m \lambda_i g_i(\mathbf{x})$$

**KKT conditions** (necessary for optimality):
1. **Stationarity**: $\nabla f(\mathbf{x}^*) = \sum_{i=1}^m \lambda_i \nabla g_i(\mathbf{x}^*)$
2. **Primal feasibility**: $g_i(\mathbf{x}^*) \geq 0$ for all $i$
3. **Dual feasibility**: $\lambda_i \geq 0$ for all $i$
4. **Complementary slackness**: $\lambda_i g_i(\mathbf{x}^*) = 0$ for all $i$

### Complementary Slackness Interpretation

For each constraint $i$:
- **If constraint is active** ($g_i(\mathbf{x}^*) = 0$): $\lambda_i$ can be positive (constraint affects optimum)
- **If constraint is inactive** ($g_i(\mathbf{x}^*) > 0$): $\lambda_i = 0$ (constraint doesn't affect optimum)

**Example for bottles**:
- Volume constraint $V \geq V_{\text{req}}$: Typically **active** ($V = V_{\text{req}}$ exactly), $\lambda_V > 0$
- Radius constraint $r_i \geq 1$ cm: May be **inactive** for most control points, $\lambda_i = 0$

## Interpreting Lagrange Multipliers

### Marginal/Shadow Price

The multiplier $\lambda$ represents **sensitivity of objective to constraint relaxation**:

$$\lambda = -\frac{dS^*}{dV_{\text{req}}}$$

(Negative sign because increasing $V_{\text{req}}$ typically increases minimum $S$)

**Physical meaning**:
- $\lambda = 0.5$ cm²/mL: If we increase target volume by 1 mL, surface area increases by approximately 0.5 cm²
- Large $\lambda$: Volume constraint is "expensive" (tight geometry)
- Small $\lambda$: Volume constraint is "cheap" (loose geometry)

### Example Calculation

Suppose:
- Optimum: $\mathbf{p}^* = [3, 4, 4.5, 4.2, 2.8, 2, 18]$ cm
- Optimal SA: $S^* = 265$ cm²
- Target volume: $V_{\text{req}} = 500$ mL
- Lagrange multiplier: $\lambda = 0.48$ cm²/mL

**Interpretation**:
1. If we increase target to 510 mL (+10 mL), SA increases by approximately $10 \times 0.48 = 4.8$ cm²
2. Predicted new SA: $265 + 4.8 = 269.8$ cm²
3. (This is a first-order approximation; exact value requires re-optimization)

### Trade-off Analysis

Plot $S^*$ vs. $V_{\text{req}}$ for different target volumes:
- Slope of this curve approximates $\lambda$
- Steep slope (large $\lambda$): Hard to add volume without increasing SA
- Shallow slope (small $\lambda$): Easy to add volume with minimal SA increase

## Numerical Implementation in SciPy

### Direct Approach (Manual Lagrangian)

Solve the system $\nabla \mathcal{L} = 0$ using `scipy.optimize.fsolve`:

```python
def lagrangian_conditions(vars):
    """
    vars = [p1, ..., pn, lambda]
    Returns residuals of KKT conditions
    """
    params = vars[:-1]
    lam = vars[-1]
    
    # Gradient of L w.r.t. params
    grad_S = numerical_gradient(surface_area, params)
    grad_V = numerical_gradient(volume, params)
    
    stationarity = grad_S - lam * grad_V  # Should be zero
    constraint = volume(params) - V_target  # Should be zero
    
    return np.concatenate([stationarity, [constraint]])

# Solve
initial_guess = np.concatenate([p0, [1.0]])  # params + lambda guess
solution = fsolve(lagrangian_conditions, initial_guess)
optimal_params = solution[:-1]
lambda_value = solution[-1]
```

### Indirect Approach (Scipy's Constrained Optimizer)

Let scipy handle Lagrange multipliers internally:

```python
from scipy.optimize import minimize

def objective(params):
    return surface_area(params)

def constraint_func(params):
    return volume(params) - V_target

constraint = {'type': 'eq', 'fun': constraint_func}

result = minimize(
    objective,
    x0=initial_params,
    method='SLSQP',
    constraints=constraint
)

optimal_params = result.x
lambda_value = result.mult['ineq'][0]  # Extract multiplier
```

**Advantage**: Scipy handles numerics, convergence, constraint satisfaction automatically.

## Advanced Topics

### Multiple Constraints

For bottle with multiple constraints:
$$
\begin{align}
\min_{\mathbf{p}} \quad & S(\mathbf{p}) \\
\text{s.t.} \quad & V(\mathbf{p}) \geq V_{\text{req}} \\
& f(x; \mathbf{p}) \geq r_{\text{min}} \quad \forall x \\
& |f'(x; \mathbf{p})| \leq s_{\text{max}} \quad \forall x
\end{align}
$$

**Lagrangian**:
$$\mathcal{L} = S - \lambda_V (V - V_{\text{req}}) - \int \lambda_r(x) (f(x) - r_{\text{min}}) dx - \int \lambda_s(x) (s_{\text{max}} - |f'(x)|) dx$$

This becomes an **infinite-dimensional** problem (continuous constraints). Discretize in practice.

### Second-Order Conditions

KKT conditions are **necessary** but not **sufficient** for optimality. 

**Sufficient condition**: Check second-order (Hessian) condition:
$$\mathbf{d}^T \nabla^2 \mathcal{L} \mathbf{d} > 0$$
for all $\mathbf{d}$ in the tangent space of active constraints.

In practice, trust that SLSQP finds local minima correctly.

### Global vs. Local Optimality

Lagrange/KKT conditions find **local extrema** only.

For non-convex problems (like ours), multiple local minima may exist:
- Cylinder-like shape (local minimum)
- Tapered shape (potentially better local minimum)
- Bulbous shape (another local minimum)

**Strategies**:
1. Multi-start: Try various initial guesses
2. Global optimization: Use `differential_evolution` first, refine with SLSQP
3. Physical intuition: Initialize with reasonable shapes (not random)

## Bottle-Specific Insights

### Expected Lagrange Multiplier Range

For typical bottles:
- Small volumes (250 mL): $\lambda \approx 0.3\text{-}0.4$ cm²/mL
- Medium volumes (500 mL): $\lambda \approx 0.4\text{-}0.5$ cm²/mL
- Large volumes (1000 mL): $\lambda \approx 0.5\text{-}0.6$ cm²/mL

**Trend**: $\lambda$ increases with volume (larger bottles have less favorable SA/V scaling).

### Active Constraints

After optimization, check which constraints are active:
```python
print("Volume constraint:", abs(V_optimal - V_target) < 1e-3)  # Usually True
print("Min radius:", min_radius_in_profile)  # Check if any point hits r_min
print("Max slope:", max_slope_in_profile)  # Check if any section hits s_max
```

If slope constraint is active, bottle is as steep as allowed → may want to relax constraint.

## Validation and Diagnostics

### Checking Optimality

1. **Gradient check**:
   ```python
   grad_S = numerical_gradient(surface_area, optimal_params)
   grad_V = numerical_gradient(volume, optimal_params)
   ratio = grad_S / grad_V  # Should be constant ≈ lambda
   print("Gradient ratio range:", ratio.min(), "-", ratio.max())
   # Expect: narrow range around lambda value
   ```

2. **Constraint satisfaction**:
   ```python
   V_actual = volume(optimal_params)
   print("Volume error:", abs(V_actual - V_target))
   # Expect: < 1 mL for 500 mL bottle
   ```

3. **Perturbation test**:
   ```python
   # Perturb solution slightly, check if objective increases
   for i in range(len(optimal_params)):
       perturbed = optimal_params.copy()
       perturbed[i] += 0.1
       SA_pert = surface_area(perturbed)
       print(f"Param {i}: SA change = {SA_pert - SA_optimal}")
       # Expect: positive (we're at minimum)
   ```

## Summary Table

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Lagrangian | $\mathcal{L} = f(\mathbf{x}) - \lambda g(\mathbf{x})$ | Combines objective and constraint |
| Optimality | $\nabla f = \lambda \nabla g$ | Gradients parallel at optimum |
| Multiplier | $\lambda = -\frac{df^*}{dg}$ | Marginal cost of constraint |
| KKT (inequality) | $\lambda g(\mathbf{x}) = 0$ | Active: $g=0, \lambda>0$; Inactive: $g>0, \lambda=0$ |

## References

1. **Textbook**: Nocedal & Wright, "Numerical Optimization" (2006), Chapters 12-14
2. **Classical reference**: Lagrange, "Mécanique Analytique" (1788) - original paper
3. **KKT conditions**: Karush (1939), Kuhn & Tucker (1951)
4. **Scipy documentation**: https://docs.scipy.org/doc/scipy/reference/optimize.html

---

**Implementation Note**: Modern optimizers like SLSQP handle Lagrange multipliers internally using KKT conditions. You specify constraints; the optimizer computes multipliers. Access them via `result.mult` after optimization completes.
