"""
Geometry Module: Surface of Revolution Calculations

This module provides functions for computing surface area and volume
of bottles modeled as surfaces of revolution, along with parametric
curve representations.
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import CubicSpline, UnivariateSpline
from typing import Callable, Tuple, List, Union


class BottleProfile:
    """
    Represents a bottle profile curve f(x) where x is height and f(x) is radius.
    
    Supports multiple parametrization types:
    - Polynomial
    - Cubic B-spline
    - Piecewise functions
    """
    
    def __init__(self, func: Callable = None, height_range: Tuple[float, float] = None,
                 derivative_func: Callable = None, name: str = "Generic",
                 radius_func: Callable = None, height: float = None):
        """
        Initialize bottle profile.
        
        Parameters
        ----------
        func : callable
            Function f(x) returning radius at height x
        height_range : tuple
            (x_min, x_max) height range in cm
        derivative_func : callable, optional
            Function f'(x) returning derivative. If None, computed numerically.
        name : str
            Descriptive name for this profile
        """
        # Backward compatibility for older notebook signatures that used
        # radius_func + height instead of func + height_range.
        if func is None and radius_func is not None:
            func = radius_func

        if height_range is None and height is not None:
            height_range = (0, float(height))

        if func is None or height_range is None:
            raise ValueError("BottleProfile requires (func and height_range) or (radius_func and height).")

        self.func = func
        self.height_range = height_range
        self.derivative_func = derivative_func
        self.name = name

    @property
    def height(self) -> float:
        """Total profile height for compatibility with existing notebooks."""
        return float(self.height_range[1] - self.height_range[0])
    
    def radius(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate radius at height(s) x"""
        return self.func(x)
    
    def derivative(self, x: Union[float, np.ndarray], h: float = 1e-5) -> Union[float, np.ndarray]:
        """
        Evaluate derivative f'(x) at height(s) x.
        
        Uses analytical derivative if provided, otherwise central difference.
        """
        if self.derivative_func is not None:
            return self.derivative_func(x)
        else:
            # Central difference: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
            return (self.func(x + h) - self.func(x - h)) / (2 * h)
    
    def is_valid(self, min_radius: float = 0.1) -> Tuple[bool, str]:
        """
        Check if profile is physically valid.
        
        Parameters
        ----------
        min_radius : float
            Minimum allowed radius in cm
        
        Returns
        -------
        valid : bool
            True if profile is valid
        message : str
            Description of validation result
        """
        x_test = np.linspace(self.height_range[0], self.height_range[1], 100)
        radii = self.radius(x_test)
        
        # Check non-negativity
        if np.any(radii < min_radius):
            return False, f"Radius drops below {min_radius} cm"
        
        # Check for NaN or Inf
        if np.any(~np.isfinite(radii)):
            return False, "Contains NaN or Inf values"
        
        # Check smoothness (no excessive slope)
        derivatives = self.derivative(x_test)
        max_slope = np.max(np.abs(derivatives))
        if max_slope > 10:
            return False, f"Excessive slope: max |f'| = {max_slope:.2f}"
        
        return True, "Valid profile"
    
    @classmethod
    def from_polynomial(cls, coeffs: List[float], height: float, name: str = "Polynomial"):
        """
        Create bottle profile from polynomial coefficients.
        
        f(x) = a0 + a1*x + a2*x^2 + ... + an*x^n
        
        Parameters
        ----------
        coeffs : list
            Polynomial coefficients [a0, a1, a2, ...]
        height : float
            Total bottle height in cm
        name : str
            Profile name
        
        Returns
        -------
        BottleProfile instance
        """
        poly = np.poly1d(coeffs[::-1])  # np.poly1d uses reverse order
        poly_derivative = np.polyder(poly)
        
        return cls(
            func=lambda x: poly(x),
            height_range=(0, height),
            derivative_func=lambda x: poly_derivative(x),
            name=name
        )
    
    @classmethod
    def from_bspline(cls, control_radii: List[float], height: float, 
                     name: str = "B-spline", bc_type: str = 'clamped'):
        """
        Create bottle profile from cubic B-spline control points.
        
        Parameters
        ----------
        control_radii : list
            Radius values at uniformly spaced heights
        height : float
            Total bottle height in cm
        name : str
            Profile name
        bc_type : str
            Boundary condition type ('clamped', 'natural', 'not-a-knot')
        
        Returns
        -------
        BottleProfile instance
        """
        n_points = len(control_radii)
        control_heights = np.linspace(0, height, n_points)
        spline = CubicSpline(control_heights, control_radii, bc_type=bc_type)
        
        return cls(
            func=lambda x: spline(x),
            height_range=(0, height),
            derivative_func=lambda x: spline(x, 1),  # First derivative
            name=name
        )
    
    @classmethod
    def cylinder(cls, radius: float, height: float):
        """Create cylindrical profile (constant radius)"""
        return cls(
            func=lambda x: radius * np.ones_like(x) if isinstance(x, np.ndarray) else radius,
            height_range=(0, height),
            derivative_func=lambda x: 0.0 if isinstance(x, float) else np.zeros_like(x),
            name=f"Cylinder(r={radius}, h={height})"
        )
    
    @classmethod
    def cone(cls, base_radius: float, top_radius: float, height: float):
        """Create conical profile (linear taper)"""
        slope = (top_radius - base_radius) / height
        return cls(
            func=lambda x: base_radius + slope * x,
            height_range=(0, height),
            derivative_func=lambda x: slope if isinstance(x, float) else np.full_like(x, slope),
            name=f"Cone(r0={base_radius}, r1={top_radius}, h={height})"
        )


def surface_area(profile: BottleProfile, method: str = 'quad', n_points: int = 1000) -> float:
    """
    Calculate surface area of bottle (surface of revolution).
    
    Formula: S = 2π ∫[0→H] f(x) * √(1 + (f'(x))²) dx
    
    Parameters
    ----------
    profile : BottleProfile
        Bottle profile curve
    method : str
        Integration method: 'quad' (adaptive) or 'simpson' (fixed grid)
    n_points : int
        Number of points for simpson method
    
    Returns
    -------
    surface_area : float
        Surface area in cm²
    """
    x_min, x_max = profile.height_range
    
    def integrand(x):
        """f(x) * sqrt(1 + f'(x)^2)"""
        f_x = profile.radius(x)
        f_prime_x = profile.derivative(x)
        return f_x * np.sqrt(1 + f_prime_x**2)
    
    if method == 'quad':
        # Adaptive quadrature (recommended for accuracy)
        area, error = quad(integrand, x_min, x_max, limit=100)
        return 2 * np.pi * area
    
    elif method == 'simpson':
        # Simpson's rule on fixed grid
        x_vals = np.linspace(x_min, x_max, n_points)
        y_vals = integrand(x_vals)
        area = simpson(y_vals, x=x_vals)
        return 2 * np.pi * area
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'quad' or 'simpson'.")


def volume(profile: BottleProfile, method: str = 'quad', n_points: int = 1000) -> float:
    """
    Calculate volume of bottle (surface of revolution).
    
    Formula: V = π ∫[0→H] [f(x)]² dx
    
    Parameters
    ----------
    profile : BottleProfile
        Bottle profile curve
    method : str
        Integration method: 'quad' (adaptive) or 'simpson' (fixed grid)
    n_points : int
        Number of points for simpson method
    
    Returns
    -------
    volume : float
        Volume in cm³ (equivalent to mL)
    """
    x_min, x_max = profile.height_range
    
    def integrand(x):
        """[f(x)]²"""
        f_x = profile.radius(x)
        return f_x**2
    
    if method == 'quad':
        vol, error = quad(integrand, x_min, x_max, limit=100)
        return np.pi * vol
    
    elif method == 'simpson':
        x_vals = np.linspace(x_min, x_max, n_points)
        y_vals = integrand(x_vals)
        vol = simpson(y_vals, x=x_vals)
        return np.pi * vol
    
    else:
        raise ValueError(f"Unknown method: {method}")


def surface_area_to_volume_ratio(profile: BottleProfile) -> float:
    """
    Calculate SA/V ratio (cm⁻¹).
    
    Lower ratio indicates more efficient shape (less surface per volume).
    """
    sa = surface_area(profile)
    vol = volume(profile)
    return sa / vol


def params_to_profile(params: np.ndarray, profile_type: str = 'bspline') -> BottleProfile:
    """
    Convert optimization parameters to BottleProfile.
    
    Parameters
    ----------
    params : ndarray
        Optimization parameters. Format depends on profile_type:
        - 'bspline': [r0, r1, ..., rn, H] where ri are control radii, H is height
        - 'polynomial': [a0, a1, ..., an, H] where ai are poly coeffs
    profile_type : str
        Type of parametrization
    
    Returns
    -------
    BottleProfile instance
    """
    if profile_type == 'bspline':
        radii = params[:-1]
        height = params[-1]
        return BottleProfile.from_bspline(radii, height)
    
    elif profile_type == 'polynomial':
        coeffs = params[:-1]
        height = params[-1]
        return BottleProfile.from_polynomial(coeffs, height)
    
    else:
        raise ValueError(f"Unknown profile type: {profile_type}")


# ============================================================================
# Validation Functions (for testing against analytical solutions)
# ============================================================================

def validate_cylinder(radius: float = 3.0, height: float = 10.0, tol: float = 1e-2):
    """
    Validate surface area and volume calculations against analytical cylinder formulas.
    
    Analytical:
        SA = 2πrh (excluding top/bottom)
        V = πr²h
    """
    profile = BottleProfile.cylinder(radius, height)
    
    # Compute
    sa_computed = surface_area(profile)
    vol_computed = volume(profile)
    
    # Analytical
    sa_analytical = 2 * np.pi * radius * height
    vol_analytical = np.pi * radius**2 * height
    
    # Compare
    sa_error = abs(sa_computed - sa_analytical) / sa_analytical
    vol_error = abs(vol_computed - vol_analytical) / vol_analytical
    
    print(f"Cylinder Validation (r={radius}, h={height}):")
    print(f"  SA:  computed={sa_computed:.4f}, analytical={sa_analytical:.4f}, error={sa_error:.2%}")
    print(f"  Vol: computed={vol_computed:.4f}, analytical={vol_analytical:.4f}, error={vol_error:.2%}")
    
    assert sa_error < tol, f"SA error {sa_error:.2%} exceeds tolerance {tol:.2%}"
    assert vol_error < tol, f"Volume error {vol_error:.2%} exceeds tolerance {tol:.2%}"
    print("  ✓ Passed\n")


def validate_cone(base_r: float = 4.0, top_r: float = 2.0, height: float = 12.0, tol: float = 1e-2):
    """
    Validate against analytical cone formulas.
    
    For truncated cone (frustum):
        SA = π(r1 + r2) * s, where s = √[h² + (r1-r2)²]
        V = (π/3) * h * (r1² + r1*r2 + r2²)
    """
    profile = BottleProfile.cone(base_r, top_r, height)
    
    # Compute
    sa_computed = surface_area(profile)
    vol_computed = volume(profile)
    
    # Analytical
    slant_height = np.sqrt(height**2 + (base_r - top_r)**2)
    sa_analytical = np.pi * (base_r + top_r) * slant_height
    vol_analytical = (np.pi / 3) * height * (base_r**2 + base_r * top_r + top_r**2)
    
    # Compare
    sa_error = abs(sa_computed - sa_analytical) / sa_analytical
    vol_error = abs(vol_computed - vol_analytical) / vol_analytical
    
    print(f"Cone Validation (r0={base_r}, r1={top_r}, h={height}):")
    print(f"  SA:  computed={sa_computed:.4f}, analytical={sa_analytical:.4f}, error={sa_error:.2%}")
    print(f"  Vol: computed={vol_computed:.4f}, analytical={vol_analytical:.4f}, error={vol_error:.2%}")
    
    assert sa_error < tol, f"SA error {sa_error:.2%} exceeds tolerance {tol:.2%}"
    assert vol_error < tol, f"Volume error {vol_error:.2%} exceeds tolerance {tol:.2%}"
    print("  ✓ Passed\n")


def validate_sphere_segment(radius: float = 5.0, height: float = 8.0, tol: float = 0.05):
    """
    Validate against spherical cap (segment) formulas.
    
    Note: Higher tolerance due to numerical integration of √(R² - (x-R)²)
    """
    # Sphere centered at (0, R) with radius R
    # Profile: f(x) = √(R² - (x - R)²) for x in [R - h, R]
    x_min = radius - height / 2
    x_max = radius + height / 2
    
    def sphere_func(x):
        val = radius**2 - (x - radius)**2
        return np.sqrt(np.maximum(val, 0))  # Avoid negative due to numerical errors
    
    profile = BottleProfile(sphere_func, (x_min, x_max), name="Sphere segment")
    
    # Compute
    sa_computed = surface_area(profile, method='quad')
    vol_computed = volume(profile, method='quad')
    
    # For a full sphere: SA = 4πR², V = (4/3)πR³
    # For segment, formulas are more complex; use numerical as reference
    print(f"Sphere Segment (R={radius}, h={height}):")
    print(f"  SA:  {sa_computed:.4f} cm²")
    print(f"  Vol: {vol_computed:.4f} cm³")
    print("  (Analytical formulas complex; visual verification needed)\n")


if __name__ == "__main__":
    print("=" * 70)
    print("Geometry Module Validation")
    print("=" * 70 + "\n")
    
    # Test 1: Cylinder
    validate_cylinder(radius=3.0, height=15.0)
    
    # Test 2: Cone
    validate_cone(base_r=4.0, top_r=2.0, height=12.0)
    
    # Test 3: B-spline bottle
    print("B-spline Bottle Example:")
    control_radii = [3.0, 4.0, 4.5, 4.0, 2.5, 2.0]
    bottle = BottleProfile.from_bspline(control_radii, height=18.0, name="Example Bottle")
    
    sa = surface_area(bottle)
    vol = volume(bottle)
    ratio = surface_area_to_volume_ratio(bottle)
    
    print(f"  Control radii: {control_radii}")
    print(f"  Height: 18.0 cm")
    print(f"  Surface Area: {sa:.2f} cm²")
    print(f"  Volume: {vol:.2f} mL")
    print(f"  SA/V ratio: {ratio:.4f} cm⁻¹")
    
    valid, msg = bottle.is_valid()
    print(f"  Validity: {msg}")
    print()
    
    # Test 4: Compare cylinder to optimized shape (conceptual)
    print("Baseline Comparison:")
    cylinder = BottleProfile.cylinder(radius=3.0, height=17.68)  # Chosen to match ~500 mL
    sa_cyl = surface_area(cylinder)
    vol_cyl = volume(cylinder)
    
    print(f"  Cylinder (r=3, h=17.68): SA={sa_cyl:.2f} cm², V={vol_cyl:.2f} mL")
    print(f"  Theoretical sphere (V=500 mL): SA={(36*np.pi*500**2)**(1/3):.2f} cm²")
    print(f"  Potential savings: {(1 - 238/sa_cyl)*100:.1f}% (if sphere were feasible)")
    print()
    
    print("=" * 70)
    print("All validations passed! Geometry module ready for optimization.")
    print("=" * 70)
