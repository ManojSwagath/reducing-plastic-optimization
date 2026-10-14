"""
Visualization Module: Plotting Functions for Bottle Profiles

Provides functions for:
- 2D profile curves
- 3D surface of revolution rendering
- Optimization landscape plots
- Trade-off curves (SA vs volume)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry import BottleProfile, surface_area, volume


def plot_profile(profile: BottleProfile, ax: Optional[plt.Axes] = None, 
                 label: Optional[str] = None, color: str = 'blue',
                 n_points: int = 200, show_derivative: bool = False) -> plt.Axes:
    """
    Plot 2D bottle profile curve.
    
    Parameters
    ----------
    profile : BottleProfile
        Bottle profile to plot
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates new figure.
    label : str, optional
        Legend label. Defaults to profile.name.
    color : str
        Line color
    n_points : int
        Number of points for smooth curve
    show_derivative : bool
        If True, plot derivative f'(x) on secondary axis
    
    Returns
    -------
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    
    x_min, x_max = profile.height_range
    x_vals = np.linspace(x_min, x_max, n_points)
    r_vals = profile.radius(x_vals)
    
    # Plot profile (showing both sides for symmetry)
    label = label or profile.name
    ax.plot(r_vals, x_vals, color=color, linewidth=2, label=label)
    ax.plot(-r_vals, x_vals, color=color, linewidth=2)
    
    # Styling
    ax.set_xlabel('Radius (cm)', fontsize=11)
    ax.set_ylabel('Height (cm)', fontsize=11)
    ax.set_title('Bottle Profile', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.legend()
    # Use box-adjustable aspect to avoid shared-axis layout errors in notebooks.
    ax.set_aspect('equal', adjustable='box')
    
    # Optional: show derivative
    if show_derivative:
        ax2 = ax.twiny()
        dr_vals = profile.derivative(x_vals)
        ax2.plot(dr_vals, x_vals, color='red', linewidth=1, 
                linestyle='--', alpha=0.6, label="f'(x)")
        ax2.set_xlabel("Slope f'(x)", fontsize=10, color='red')
        ax2.tick_params(axis='x', labelcolor='red')
        ax2.legend(loc='upper right')
    
    return ax


def plot_bottle_3d(profile: BottleProfile, ax: Optional[Axes3D] = None,
                   n_theta: int = 40, n_x: int = 100, 
                   color_map: str = 'viridis', alpha: float = 0.8,
                   show_wireframe: bool = False) -> Axes3D:
    """
    Plot 3D surface of revolution for bottle.
    
    Parameters
    ----------
    profile : BottleProfile
        Bottle profile to render
    ax : Axes3D, optional
        3D axes to plot on. If None, creates new figure.
    n_theta : int
        Number of angular divisions (resolution around axis)
    n_x : int
        Number of height divisions (resolution along axis)
    color_map : str
        Matplotlib colormap name
    alpha : float
        Surface transparency (0=transparent, 1=opaque)
    show_wireframe : bool
        If True, overlay wireframe on surface
    
    Returns
    -------
    ax : Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Generate mesh
    x_min, x_max = profile.height_range
    x_vals = np.linspace(x_min, x_max, n_x)
    theta_vals = np.linspace(0, 2 * np.pi, n_theta)
    
    X, Theta = np.meshgrid(x_vals, theta_vals)
    R = profile.radius(X)
    
    # Convert to Cartesian coordinates
    Y = R * np.cos(Theta)
    Z = R * np.sin(Theta)
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, cmap=color_map, alpha=alpha, 
                           edgecolor='none', shade=True)
    
    # Optional wireframe
    if show_wireframe:
        ax.plot_wireframe(X, Y, Z, color='black', linewidth=0.3, alpha=0.3)
    
    # Styling
    ax.set_xlabel('Height (cm)', fontsize=10)
    ax.set_ylabel('X (cm)', fontsize=10)
    ax.set_zlabel('Y (cm)', fontsize=10)
    ax.set_title(f'Bottle 3D: {profile.name}', fontsize=12, fontweight='bold')
    
    # Equal aspect ratio for circular cross-sections
    max_range = max(x_max - x_min, 2 * max(profile.radius(x_vals)))
    ax.set_box_aspect([1, 1, 1])
    
    return ax


def plot_multiple_profiles(profiles: List[BottleProfile], 
                          title: str = "Bottle Profile Comparison",
                          colors: Optional[List[str]] = None) -> plt.Figure:
    """
    Plot multiple bottle profiles side-by-side for comparison.
    
    Parameters
    ----------
    profiles : list of BottleProfile
        Profiles to compare
    title : str
        Figure title
    colors : list of str, optional
        Colors for each profile. If None, uses default color cycle.
    
    Returns
    -------
    fig : matplotlib Figure
    """
    n_profiles = len(profiles)
    fig, axes = plt.subplots(1, n_profiles, figsize=(4 * n_profiles, 8), 
                             sharey=True, sharex=True)
    
    if n_profiles == 1:
        axes = [axes]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_profiles))
    
    for ax, prof, col in zip(axes, profiles, colors):
        plot_profile(prof, ax=ax, color=col)
        sa = surface_area(prof)
        vol = volume(prof)
        ax.set_title(f"{prof.name}\nSA={sa:.1f} cm², V={vol:.1f} mL", 
                    fontsize=10)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig


def plot_optimization_landscape(objective_func, param_ranges: List[Tuple[float, float]], 
                                param_names: List[str], 
                                fixed_params: Optional[np.ndarray] = None,
                                n_points: int = 50, optimum: Optional[np.ndarray] = None) -> plt.Figure:
    """
    Plot 2D contour of objective function over parameter space.
    
    Useful for visualizing optimization landscape and sensitivity.
    
    Parameters
    ----------
    objective_func : callable
        Function to evaluate, takes full parameter vector
    param_ranges : list of tuples
        [(min, max)] for each of 2 parameters to vary
    param_names : list of str
        Names for the 2 parameters
    fixed_params : ndarray, optional
        Values for other parameters (if objective requires >2 params)
    n_points : int
        Grid resolution
    optimum : ndarray, optional
        Location of optimum (marked with red X)
    
    Returns
    -------
    fig : matplotlib Figure
    """
    if len(param_ranges) != 2 or len(param_names) != 2:
        raise ValueError("Must provide exactly 2 parameters for 2D contour plot")
    
    # Create grid
    p1_vals = np.linspace(*param_ranges[0], n_points)
    p2_vals = np.linspace(*param_ranges[1], n_points)
    P1, P2 = np.meshgrid(p1_vals, p2_vals)
    
    # Evaluate objective
    Z = np.zeros_like(P1)
    for i in range(n_points):
        for j in range(n_points):
            if fixed_params is not None:
                params = fixed_params.copy()
                params[0] = P1[i, j]
                params[1] = P2[i, j]
            else:
                params = np.array([P1[i, j], P2[i, j]])
            
            try:
                Z[i, j] = objective_func(params)
            except:
                Z[i, j] = np.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    
    levels = 20
    contour = ax.contourf(P1, P2, Z, levels=levels, cmap='viridis', alpha=0.8)
    ax.contour(P1, P2, Z, levels=levels, colors='black', linewidths=0.5, alpha=0.3)
    
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Objective Function Value', fontsize=11)
    
    # Mark optimum
    if optimum is not None:
        ax.plot(optimum[0], optimum[1], 'rx', markersize=15, markeredgewidth=3,
               label='Optimum')
        ax.legend()
    
    ax.set_xlabel(param_names[0], fontsize=12)
    ax.set_ylabel(param_names[1], fontsize=12)
    ax.set_title('Optimization Landscape', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_tradeoff_curve(volumes: np.ndarray, surface_areas: np.ndarray,
                       labels: Optional[List[str]] = None,
                       highlight_indices: Optional[List[int]] = None) -> plt.Figure:
    """
    Plot surface area vs volume trade-off curve.
    
    Parameters
    ----------
    volumes : ndarray
        Volume values (mL)
    surface_areas : ndarray
        Corresponding surface areas (cm²)
    labels : list of str, optional
        Labels for each point
    highlight_indices : list of int, optional
        Indices of points to highlight
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Main curve
    ax.plot(volumes, surface_areas, 'o-', linewidth=2, markersize=8, 
           color='steelblue', label='Optimized shapes')
    
    # Highlight specific points
    if highlight_indices is not None:
        v_highlight = volumes[highlight_indices]
        sa_highlight = surface_areas[highlight_indices]
        ax.plot(v_highlight, sa_highlight, 'ro', markersize=12, 
               markeredgewidth=2, markerfacecolor='none', label='Selected designs')
    
    # Add labels
    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (volumes[i], surface_areas[i]), 
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, alpha=0.8)
    
    # Theoretical bounds
    v_range = np.linspace(volumes.min(), volumes.max(), 100)
    sa_sphere = (36 * np.pi * v_range**2)**(1/3)
    ax.plot(v_range, sa_sphere, '--', color='green', linewidth=2, 
           alpha=0.6, label='Sphere (theoretical optimum)')
    
    # Styling
    ax.set_xlabel('Volume (mL)', fontsize=12)
    ax.set_ylabel('Surface Area (cm²)', fontsize=12)
    ax.set_title('Surface Area vs Volume Trade-off', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    return fig


def plot_convergence(iteration_history: List[float], ylabel: str = "Objective Value") -> plt. Figure:
    """
    Plot optimization convergence history.
    
    Parameters
    ----------
    iteration_history : list
        Objective function values at each iteration
    ylabel : str
        Label for y-axis
    
    Returns
    -------
    fig : matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    iterations = np.arange(len(iteration_history))
    ax.plot(iterations, iteration_history, 'o-', linewidth=2, markersize=5, color='darkblue')
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add final value annotation
    final_val = iteration_history[-1]
    ax.annotate(f'Final: {final_val:.2f}', 
               xy=(len(iteration_history)-1, final_val),
               xytext=(-50, 20), textcoords='offset points',
               fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    return fig


def plot_profile_comparison_table(profiles: List[BottleProfile]) -> plt.Figure:
    """
    Create comparison table showing SA, V, and SA/V for multiple profiles.
    
    Parameters
    ----------
    profiles : list of BottleProfile
        Profiles to compare
    
    Returns
    -------
    fig : matplotlib Figure
    """
    # Calculate metrics
    names = []
    sas = []
    vols = []
    ratios = []
    
    for prof in profiles:
        names.append(prof.name)
        sa = surface_area(prof)
        vol = volume(prof)
        sas.append(sa)
        vols.append(vol)
        ratios.append(sa / vol)
    
    # Create table
    fig, ax = plt.subplots(figsize=(10, len(profiles) * 0.6 + 1))
    ax.axis('off')
    
    table_data = []
    for i in range(len(profiles)):
        table_data.append([
            names[i],
            f"{sas[i]:.2f}",
            f"{vols[i]:.2f}",
            f"{ratios[i]:.4f}"
        ])
    
    table = ax.table(cellText=table_data,
                    colLabels=['Profile', 'SA (cm²)', 'Volume (mL)', 'SA/V (cm⁻¹)'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.4, 0.2, 0.2, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(profiles) + 1):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax.set_title('Bottle Profile Comparison', fontsize=14, fontweight='bold', pad=20)
    
    return fig


if __name__ == "__main__":
    print("Testing visualization module...")
    
    # Create test profiles
    cylinder = BottleProfile.cylinder(3.0, 15.0)
    cone = BottleProfile.cone(4.0, 2.0, 15.0)
    bspline = BottleProfile.from_bspline([3, 4, 4.5, 4, 2.5, 2], height=18)
    
    profiles = [cylinder, cone, bspline]
    
    # Test 1: Single profile 2D
    print("1. Plotting 2D profile...")
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plot_profile(bspline, ax=ax1, show_derivative=True)
    plt.savefig('test_profile_2d.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_profile_2d.png")
    
    # Test 2: 3D bottle
    print("2. Plotting 3D bottle...")
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    plot_bottle_3d(bspline, ax=ax2)
    plt.savefig('test_bottle_3d.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_bottle_3d.png")
    
    # Test 3: Multiple profiles
    print("3. Plotting profile comparison...")
    fig3 = plot_multiple_profiles(profiles, title="Shape Comparison")
    plt.savefig('test_comparison.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_comparison.png")
    
    # Test 4: Trade-off curve
    print("4. Plotting trade-off curve...")
    volumes_test = np.array([250, 500, 750, 1000])
    sas_test = np.array([180, 265, 340, 410])
    fig4 = plot_tradeoff_curve(volumes_test, sas_test, 
                              labels=[f"{v}mL" for v in volumes_test])
    plt.savefig('test_tradeoff.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_tradeoff.png")
    
    # Test 5: Comparison table
    print("5. Creating comparison table...")
    fig5 = plot_profile_comparison_table(profiles)
    plt.savefig('test_table.png', dpi=100, bbox_inches='tight')
    print("   Saved: test_table.png")
    
    print("\nAll visualization tests complete!")
    print("Check generated PNG files for results.")
