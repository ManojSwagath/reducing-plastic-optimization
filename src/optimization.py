"""
Optimization Module: Bottle Shape Optimization

Implements constrained optimization for bottle shapes:
- Minimize surface area
- Subject to volume constraints
- Using SLSQP (Sequential Least Squares Programming)
- Extract and analyze Lagrange multipliers
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry import BottleProfile, surface_area, volume, params_to_profile


class OptimizationResult:
    """Container for optimization results"""
    
    def __init__(self, success: bool, optimal_params: np.ndarray,
                 final_surface_area: float, final_volume: float,
                 lagrange_multipliers: Optional[np.ndarray],
                 num_iterations: int, convergence_history: List[float],
                 profile: BottleProfile, message: str):
        self.success = success
        self.optimal_params = optimal_params
        self.final_surface_area = final_surface_area
        self.final_volume = final_volume
        self.lagrange_multipliers = lagrange_multipliers
        self.num_iterations = num_iterations
        self.convergence_history = convergence_history
        self.profile = profile
        self.message = message
    
    def __repr__(self):
        status = "SUCCESS" if self.success else "FAILED"
        return (f"OptimizationResult({status}, SA={self.final_surface_area:.2f} cm², "
                f"V={self.final_volume:.2f} mL, λ={self.lagrange_multipliers})")
    
    def summary(self) -> str:
        """Return formatted summary of optimization results"""
        lines = [
            "=" * 70,
            "OPTIMIZATION RESULTS",
            "=" * 70,
            f"Status: {'SUCCESS ✓' if self.success else 'FAILED ✗'}",
            f"Message: {self.message}",
            "",
            "Final Design:",
            f"  Surface Area: {self.final_surface_area:.2f} cm²",
            f"  Volume:       {self.final_volume:.2f} mL",
            f"  SA/V Ratio:   {self.final_surface_area / self.final_volume:.4f} cm⁻¹",
            "",
            f"Optimization Details:",
            f"  Iterations:   {self.num_iterations}",
            f"  Parameters:   {self.optimal_params}",
        ]
        
        if self.lagrange_multipliers is not None:
            lines.append(f"  Lagrange mult: {self.lagrange_multipliers}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


def optimize_bottle(target_volume: float,
                   initial_params: Optional[np.ndarray] = None,
                   profile_type: str = 'bspline',
                   n_control_points: int = 6,
                   method: str = 'SLSQP',
                   bounds_radii: Optional[Tuple[float, float]] = None,
                   bounds: Optional[List[Tuple[float, float]]] = None,
                   constraints_dict: Optional[Dict] = None,
                   max_iterations: int = 200,
                   tolerance: float = 1e-6,
                   verbose: bool = True) -> OptimizationResult:
    """
    Optimize bottle shape to minimize surface area for given volume.
    
    Parameters
    ----------
    target_volume : float
        Required bottle volume in mL (cm³)
    initial_params : ndarray, optional
        Initial guess for parameters. If None, uses cylindrical starting point.
    profile_type : str
        'bspline' or 'polynomial'
    n_control_points : int
        Number of control points for B-spline (ignored for polynomial)
    method : str
        Scipy optimization method ('SLSQP', 'trust-constr', 'COBYLA')
    bounds : list of tuples, optional
        Parameter bounds [(min, max), ...]. If None, uses default bounds.
    bounds_radii : tuple, optional
        Convenience radius bounds (r_min, r_max) for B-spline profiles.
        Used only when explicit bounds are not supplied.
    constraints_dict : dict, optional
        Additional custom constraints
    max_iterations : int
        Maximum number of optimization iterations
    tolerance : float
        Convergence tolerance
    verbose : bool
        Print progress messages
    
    Returns
    -------
    OptimizationResult object containing optimal design and statistics
    """
    
    # Track convergence history
    history = []
    
    def objective(params):
        """Surface area to minimize"""
        try:
            prof = params_to_profile(params, profile_type)
            sa = surface_area(prof, method='quad')
            history.append(sa)
            return sa
        except Exception as e:
            if verbose:
                print(f"Warning: Objective evaluation failed: {e}")
            return 1e6  # Large penalty for invalid profiles
    
    def volume_constraint(params):
        """Volume constraint: V(params) - V_target >= 0"""
        try:
            prof = params_to_profile(params, profile_type)
            vol = volume(prof, method='quad')
            return vol - target_volume
        except Exception as e:
            if verbose:
                print(f"Warning: Volume constraint evaluation failed: {e}")
            return -target_volume  # Violated constraint
    
    def volume_constraint_equality(params):
        """Equality constraint for volume"""
        return abs(volume_constraint(params))
    
    # Set up initial parameters
    if initial_params is None:
        # Start with a reasonable cylindrical-like shape
        # Estimate radius assuming cylinder: V = π r² h, choose h/r ≈ 5
        h_guess = (5 * target_volume / np.pi)**(1/3) * 5
        r_guess = np.sqrt(target_volume / (np.pi * h_guess))
        
        if profile_type == 'bspline':
            # Create tapering profile: wide at base, narrow at top
            radii = np.linspace(r_guess * 1.1, r_guess * 0.7, n_control_points)
            initial_params = np.append(radii, h_guess)
        else:
            # Polynomial [a0, a1, a2, a3, H]
            initial_params = np.array([r_guess, 0, 0, 0, h_guess])
    
    # Set up bounds
    if bounds is None:
        if profile_type == 'bspline':
            # Radius bounds: 1-6 cm, height: 10-30 cm
            if bounds_radii is None:
                radius_min, radius_max = 1.0, 6.0
            else:
                radius_min, radius_max = bounds_radii
            radius_bounds = [(radius_min, radius_max)] * n_control_points
            height_bounds = [(10.0, 30.0)]
            bounds = radius_bounds + height_bounds
        else:
            # Polynomial coefficients + height
            bounds = [(1, 6), (-1, 1), (-0.1, 0.1), (-0.01, 0.01), (10, 30)]
    
    # Set up constraints
    constraints = []
    
    if method in ['SLSQP', 'trust-constr']:
        # Use inequality constraint: V >= V_target
        constraints.append({
            'type': 'ineq',
            'fun': volume_constraint
        })
    elif method == 'COBYLA':
        # COBYLA only handles inequalities
        constraints.append({
            'type': 'ineq',
            'fun': volume_constraint
        })
    
    # Add custom constraints if provided
    if constraints_dict is not None:
        constraints.extend(constraints_dict)
    
    # Run optimization
    if verbose:
        print(f"Starting optimization...")
        print(f"  Target volume: {target_volume:.0f} mL")
        print(f"  Initial params: {initial_params}")
        print(f"  Method: {method}")
        print(f"  Max iterations: {max_iterations}")
        print()
    
    result = minimize(
        objective,
        x0=initial_params,
        method=method,
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': max_iterations,
            'ftol': tolerance,
            'disp': verbose
        }
    )
    
    # Extract results
    optimal_params = result.x
    final_sa = result.fun
    final_profile = params_to_profile(optimal_params, profile_type)
    final_volume = volume(final_profile, method='quad')
    
    # Extract Lagrange multipliers (if available)
    lagrange_mult = None
    if hasattr(result, 'v') and result.v is not None:
        lagrange_mult = result.v  # For trust-constr
    elif hasattr(result, 'mult'):
        lagrange_mult = result.mult  # For SLSQP (not always populated)
    
    # Create result object
    opt_result = OptimizationResult(
        success=result.success,
        optimal_params=optimal_params,
        final_surface_area=final_sa,
        final_volume=final_volume,
        lagrange_multipliers=lagrange_mult,
        num_iterations=result.nit if hasattr(result, 'nit') else len(history),
        convergence_history=history,
        profile=final_profile,
        message=result.message
    )
    
    if verbose:
        print(opt_result.summary())
    
    return opt_result


def multi_volume_optimization(volumes: List[float],
                              profile_type: str = 'bspline',
                              n_control_points: int = 6,
                              method: str = 'SLSQP',
                              verbose: bool = False) -> List[OptimizationResult]:
    """
    Optimize bottles for multiple volume targets.
    
    Useful for generating trade-off curves (SA vs V).
    
    Parameters
    ----------
    volumes : list of float
        Target volumes in mL
    profile_type : str
        Profile parametrization
    n_control_points : int
        Number of control points
    method : str
        Optimization method
    verbose : bool
        Print per-optimization details
    
    Returns
    -------
    results : list of OptimizationResult
    """
    results = []
    
    print(f"Optimizing for {len(volumes)} volume targets...")
    print("=" * 70)
    
    for i, vol in enumerate(volumes):
        print(f"\n[{i+1}/{len(volumes)}] Target: {vol:.0f} mL")
        result = optimize_bottle(
            target_volume=vol,
            profile_type=profile_type,
            n_control_points=n_control_points,
            method=method,
            verbose=verbose
        )
        results.append(result)
        
        if result.success:
            print(f"  ✓ SA = {result.final_surface_area:.2f} cm², "
                  f"V = {result.final_volume:.2f} mL")
        else:
            print(f"  ✗ Failed: {result.message}")
    
    print("\n" + "=" * 70)
    print("Batch optimization complete!")
    
    return results


def compute_lagrange_multiplier_numerical(result: OptimizationResult,
                                         delta_volume: float = 1.0) -> float:
    """
    Numerically estimate Lagrange multiplier by finite difference.
    
    λ ≈ dSA/dV at optimal point
    
    Parameters
    ----------
    result : OptimizationResult
        Optimization result at target volume V
    delta_volume : float
        Small volume perturbation for finite difference
    
    Returns
    -------
    lambda_estimate : float
        Estimated Lagrange multiplier (cm²/mL)
    """
    V_base = result.final_volume
    SA_base = result.final_surface_area
    
    # Optimize for slightly larger volume
    result_plus = optimize_bottle(
        target_volume=V_base + delta_volume,
        initial_params=result.optimal_params,
        verbose=False
    )
    
    if not result_plus.success:
        print("Warning: Perturbed optimization failed")
        return np.nan
    
    # Finite difference
    dSA = result_plus.final_surface_area - SA_base
    dV = result_plus.final_volume - V_base
    
    lambda_est = dSA / dV
    
    return lambda_est


def sensitivity_analysis(result: OptimizationResult,
                        param_index: int,
                        perturbation: float = 0.1) -> Dict:
    """
    Analyze sensitivity of objective to parameter perturbation.
    
    Parameters
    ----------
    result : OptimizationResult
        Optimal result
    param_index : int
        Index of parameter to perturb
    perturbation : float
        Magnitude of perturbation
    
    Returns
    -------
    sensitivity_info : dict
        Contains SA change, volume change, validity
    """
    params_base = result.optimal_params.copy()
    
    # Perturb parameter
    params_pert = params_base.copy()
    params_pert[param_index] += perturbation
    
    # Evaluate
    try:
        prof_pert = params_to_profile(params_pert, 'bspline')
        sa_pert = surface_area(prof_pert)
        vol_pert = volume(prof_pert)
        
        return {
            'param_index': param_index,
            'perturbation': perturbation,
            'SA_change': sa_pert - result.final_surface_area,
            'volume_change': vol_pert - result.final_volume,
            'SA_gradient': (sa_pert - result.final_surface_area) / perturbation,
            'valid': True
        }
    except Exception as e:
        return {
            'param_index': param_index,
            'perturbation': perturbation,
            'valid': False,
            'error': str(e)
        }


if __name__ == "__main__":
    print("=" * 70)
    print("Testing Optimization Module")
    print("=" * 70 + "\n")
    
    # Test 1: Single optimization
    print("Test 1: Optimize for 500 mL bottle\n")
    result = optimize_bottle(
        target_volume=500,
        n_control_points=6,
        method='SLSQP',
        verbose=True
    )
    
    print(f"\nConvergence: {len(result.convergence_history)} evaluations")
    print(f"Volume error: {abs(result.final_volume - 500):.2f} mL")
    print(f"Profile valid: {result.profile.is_valid()}")
    
    # Test 2: Multi-volume optimization
    print("\n\nTest 2: Optimize for multiple volumes\n")
    volumes = [250, 500, 750, 1000]
    results = multi_volume_optimization(volumes, verbose=False)
    
    print("\nResults Summary:")
    print(f"{'Volume (mL)':<15} {'SA (cm²)':<15} {'SA/V':<15} {'Status'}")
    print("-" * 60)
    for vol, res in zip(volumes, results):
        status = "✓" if res.success else "✗"
        print(f"{vol:<15.0f} {res.final_surface_area:<15.2f} "
              f"{res.final_surface_area/res.final_volume:<15.4f} {status}")
    
    # Test 3: Lagrange multiplier estimation
    print("\n\nTest 3: Lagrange multiplier estimation\n")
    lambda_est = compute_lagrange_multiplier_numerical(result, delta_volume=10)
    print(f"Estimated λ: {lambda_est:.4f} cm²/mL")
    print(f"Interpretation: Adding 1 mL increases SA by ~{lambda_est:.4f} cm²")
    
    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
