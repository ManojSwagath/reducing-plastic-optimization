# Surface of Revolution: Mathematical Foundation

## Overview

A **surface of revolution** is a 3D surface created by rotating a 2D curve about an axis. For bottle design, we rotate a profile curve $f(x)$ about the vertical (height) axis to generate an axisymmetric bottle shape.

## Geometric Setup

### Coordinate System
- **Vertical axis**: $x$ (height), ranging from $x = 0$ (base) to $x = H$ (top)
- **Radial distance**: $f(x)$ represents the radius of the bottle at height $x$
- **Rotation**: Full $2\pi$ rotation about the $x$-axis

### Profile Curve Constraints
For a physically realizable bottle:
1. **Non-negativity**: $f(x) \geq 0$ for all $x \in [0, H]$
2. **Smoothness**: $f(x)$ should be at least $C^1$ continuous (no sharp corners)
3. **Boundary conditions**: 
   - $f(0) = r_{\text{base}}$ (base radius)
   - $f(H) = r_{\text{top}}$ (opening radius)

## Surface Area Formula

### Derivation Intuition

Consider an infinitesimal segment of the curve $f(x)$ between $x$ and $x + dx$:
- The segment has radial position $f(x)$
- The arc length of this segment is $ds = \sqrt{1 + (f'(x))^2} \, dx$
- When rotated $2\pi$ radians, this segment traces a circular band
- The circumference of this band is $2\pi f(x)$
- The area of the band is (circumference) × (arc length)

### Formula

The total surface area of the bottle (excluding top/bottom caps) is:

$$S = \int_0^H 2\pi f(x) \cdot ds = 2\pi \int_0^H f(x) \sqrt{1 + \left(\frac{df}{dx}\right)^2} \, dx$$

where:
- $f(x)$ is the radius at height $x$
- $f'(x) = \frac{df}{dx}$ is the derivative (slope) of the profile
- The term $\sqrt{1 + (f')^2}$ accounts for the tilt of the surface

### Special Cases

**Cylinder** ($f(x) = r$ constant):
- $f'(x) = 0$
- $S = 2\pi r \int_0^H dx = 2\pi r H$

**Cone** ($f(x) = r_0 - \frac{r_0}{H}x$, tapering from $r_0$ to $0$):
- $f'(x) = -\frac{r_0}{H}$
- $S = 2\pi r_0 \int_0^H \left(1 - \frac{x}{H}\right) \sqrt{1 + \frac{r_0^2}{H^2}} \, dx = \pi r_0 \sqrt{r_0^2 + H^2}$

**Sphere** ($f(x) = \sqrt{R^2 - (x - R)^2}$ for $x \in [0, 2R]$):
- $S = 4\pi R^2$ (well-known result)

## Volume Formula

### Derivation Intuition

At height $x$, a thin horizontal slice (disk) has:
- Radius: $f(x)$
- Thickness: $dx$
- Area: $\pi [f(x)]^2$
- Volume: $\pi [f(x)]^2 dx$

Summing (integrating) all slices from $x = 0$ to $x = H$:

### Formula

$$V = \pi \int_0^H [f(x)]^2 \, dx$$

### Special Cases

**Cylinder** ($f(x) = r$ constant):
$$V = \pi r^2 H$$

**Cone** ($f(x) = r_0(1 - x/H)$):
$$V = \pi \int_0^H r_0^2 \left(1 - \frac{x}{H}\right)^2 dx = \frac{1}{3}\pi r_0^2 H$$

**Sphere** ($f(x) = \sqrt{R^2 - (x-R)^2}$):
$$V = \frac{4}{3}\pi R^3$$

## Numerical Computation

In practice, we compute these integrals numerically using:

1. **Scipy's `quad` function** (adaptive quadrature):
   - Automatically adjusts sampling density
   - Handles complex integrands with $\sqrt{1 + (f')^2}$ term

2. **Finite difference for derivatives**:
   - $f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}$ (central difference)
   - Critical for surface area calculations

3. **Validation**:
   - Always test against analytical solutions (cylinder, cone, sphere)
   - Check convergence with increasing mesh resolution

## Geometric Insights

### Surface Area Sensitivity

The term $\sqrt{1 + (f')^2}$ in the surface area formula reveals:
- **Vertical surfaces** ($f' \approx 0$): Minimal SA, just $2\pi f(x) \cdot dx$
- **Tilted surfaces** ($|f'| > 0$): Increased SA by factor $\sqrt{1 + (f')^2}$
- **Sharp transitions** ($|f'|$ large): Dramatically increase surface area

**Design implication**: Smooth, gradually tapered shapes minimize surface area.

### Isoperimetric Connection

The classic **isoperimetric problem** asks: "What shape encloses maximum volume for given perimeter?"
- In 2D: Circle is optimal
- In 3D: Sphere is optimal

Our problem is the dual:
- **Minimize surface area for given volume**
- Sphere is still theoretically optimal ($S = (36\pi V^2)^{1/3}$)
- But spheres are impractical as bottles (unstable, no opening)
- Constraints force sub-optimal but practical shapes

## Implementation Notes

### Numerical Stability

1. **Avoid steep slopes**: Constrain $|f'(x)| < 3$ to prevent $\sqrt{1 + (f')^2}$ from becoming ill-conditioned
2. **Positive radius**: Use parametrizations that guarantee $f(x) > 0$ (e.g., $f(x) = e^{g(x)}$)
3. **Adaptive integration**: Let scipy choose sampling points, don't use fixed uniform mesh

### Parametrization Choices

For optimization, represent $f(x)$ as:
- **Polynomials**: $f(x) = a_0 + a_1 x + a_2 x^2 + \ldots$ (simple, limited flexibility)
- **B-splines**: Cubic splines with $N$ control points (recommended, $C^2$ smooth)
- **Piecewise**: Different functions for base, body, neck (requires continuity enforcement)

## References

1. **Calculus textbook**: Stewart, "Calculus: Early Transcendentals", Chapter 6.3 (Volumes by Cylindrical Shells)
2. **Differential Geometry**: do Carmo, "Differential Geometry of Curves and Surfaces", Chapter 3
3. **Numerical Integration**: Press et al., "Numerical Recipes", Chapter 4

---

**Next**: [Optimization Formulation](optimization_formulation.md)
