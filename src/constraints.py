"""
Constraints Module: Learning Feasibility Constraints from Real Bottle Data

Uses machine learning (regression, clustering) to extract design constraints
from real bottle profiles:
- Slope bounds (manufacturability)
- Radius bounds at different heights
- Geometric ratios (height/width)
- Structural integrity thresholds
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Optional
import os
import sys
import json

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from geometry import BottleProfile, surface_area, volume


class BottleConstraintLearner:
    """
    Learn feasibility constraints from real bottle profile data.
    
    Extracts:
    - Maximum slope |f'(x)|
    - Minimum neck radius
    - Height-to-width ratio bounds
    - Surface area lower bounds per volume
    """
    
    def __init__(self, data_directory: str = 'data/raw_profiles'):
        """
        Initialize constraint learner.
        
        Parameters
        ----------
        data_directory : str
            Directory containing CSV bottle profile files
        """
        self.data_directory = data_directory
        self.profiles = []
        self.metadata = {}
        self.constraints = {}

    def __contains__(self, key: str) -> bool:
        """Dictionary-style key checks for notebook compatibility."""
        return key in self._constraints_view()

    def __getitem__(self, key: str):
        """Dictionary-style access for notebook compatibility."""
        return self._constraints_view()[key]

    def keys(self):
        return self._constraints_view().keys()

    def items(self):
        return self._constraints_view().items()

    def _constraints_view(self) -> Dict:
        """Return normalized constraint keys expected by analysis notebooks."""
        view = dict(self.constraints)

        if 'slopes' in self.constraints:
            s = self.constraints['slopes']
            view['slope_constraints'] = {
                'max_slope': s['max_slope_observed'],
                'slope_95th': s['max_slope_95th_percentile'],
                'recommended_bound': s['max_slope_95th_percentile']
            }

        if 'radius_bounds' in self.constraints:
            sections = []
            for key, val in self.constraints['radius_bounds'].items():
                pct = float(key.replace('height_', '').replace('pct', ''))
                sections.append({
                    'height_percentile': pct,
                    'min_radius': val['min'],
                    'max_radius': val['max'],
                    'mean_radius': val['mean']
                })
            sections.sort(key=lambda x: x['height_percentile'])
            view['radius_constraints'] = sections

        if 'sa_model' in self.constraints:
            view['sa_threshold'] = self.constraints['sa_model']

        if 'clusters' in self.constraints:
            view['bottle_clusters'] = self.constraints['clusters']

        return view
        
    def load_bottle_profiles(self, data_directory: Optional[str] = None) -> int:
        """
        Load all bottle profiles from CSV files.
        
        Returns
        -------
        n_loaded : int
            Number of profiles successfully loaded
        """
        if data_directory is not None:
            self.data_directory = str(data_directory)

        # Allow repeated calls without duplicating entries.
        self.profiles = []
        self.metadata = {}

        if not os.path.exists(self.data_directory):
            raise FileNotFoundError(f"Data directory not found: {self.data_directory}")
        
        csv_files = [f for f in os.listdir(self.data_directory) if f.endswith('.csv')]
        
        print(f"Loading bottle profiles from {self.data_directory}...")
        
        for csv_file in csv_files:
            filepath = os.path.join(self.data_directory, csv_file)
            df = pd.read_csv(filepath)
            
            # Create BottleProfile from data
            heights = df['height_cm'].values
            radii = df['radius_cm'].values
            
            # Interpolate to create smooth profile
            from scipy.interpolate import interp1d
            interp_func = interp1d(heights, radii, kind='cubic', 
                                  bounds_error=False, fill_value='extrapolate')
            
            H = heights.max()
            profile = BottleProfile(
                func=interp_func,
                height_range=(0, H),
                name=csv_file.replace('.csv', '')
            )
            
            self.profiles.append(profile)
            
            # Compute metrics
            sa = surface_area(profile)
            vol = volume(profile)
            
            self.metadata[csv_file] = {
                'name': csv_file,
                'height': H,
                'max_radius': radii.max(),
                'min_radius': radii.min(),
                'surface_area': sa,
                'volume': vol,
                'sa_v_ratio': sa / vol
            }
        
        print(f"Loaded {len(self.profiles)} bottle profiles")
        return len(self.profiles)
    
    def compute_slope_constraints(self) -> Dict[str, float]:
        """
        Compute maximum feasible slope from real bottles.
        
        Returns
        -------
        slope_bounds : dict
            {'max_slope': value, 'percentile_95': value}
        """
        max_slopes = []
        
        for profile in self.profiles:
            x_vals = np.linspace(*profile.height_range, 100)
            derivatives = profile.derivative(x_vals)
            max_slope = np.max(np.abs(derivatives))
            max_slopes.append(max_slope)
        
        constraints = {
            'max_slope_observed': np.max(max_slopes),
            'max_slope_95th_percentile': np.percentile(max_slopes, 95),
            'max_slope_median': np.median(max_slopes),
            'max_slope_mean': np.mean(max_slopes)
        }
        
        self.constraints['slopes'] = constraints
        
        print(f"\nSlope Constraints:")
        print(f"  Max observed: {constraints['max_slope_observed']:.3f}")
        print(f"  95th percentile: {constraints['max_slope_95th_percentile']:.3f}")
        print(f"  Recommended bound: |f'(x)| <= {constraints['max_slope_95th_percentile']:.2f}")
        
        return constraints
    
    def compute_radius_constraints(self, n_height_sections: int = 5) -> List[Dict]:
        """
        Compute radius bounds at different height percentiles.
        
        Parameters
        ----------
        n_height_sections : int
            Number of height sections to analyze
        
        Returns
        -------
        radius_bounds : dict
            Bounds at different relative heights
        """
        height_percentiles = np.linspace(0, 100, n_height_sections)
        bounds = {}
        
        for pct in height_percentiles:
            radii_at_pct = []
            
            for profile in self.profiles:
                H = profile.height_range[1]
                h = (pct / 100) * H
                r = profile.radius(h)
                radii_at_pct.append(r)
            
            bounds[f'height_{int(pct)}pct'] = {
                'min': np.min(radii_at_pct),
                'max': np.max(radii_at_pct),
                'mean': np.mean(radii_at_pct),
                'std': np.std(radii_at_pct)
            }
        
        self.constraints['radius_bounds'] = bounds
        
        print(f"\nRadius Constraints at Different Heights:")
        for key, val in bounds.items():
            print(f"  {key}: {val['min']:.2f} - {val['max']:.2f} cm "
                  f"(mean: {val['mean']:.2f})")
        
        sections = []
        for key, val in bounds.items():
            pct = float(key.replace('height_', '').replace('pct', ''))
            sections.append({
                'height_percentile': pct,
                'min_radius': val['min'],
                'max_radius': val['max'],
                'mean_radius': val['mean']
            })
        sections.sort(key=lambda x: x['height_percentile'])

        return sections
    
    def compute_geometric_ratios(self) -> Dict[str, Dict[str, float]]:
        """
        Compute height-to-width and other geometric ratio constraints.

        Returns
        -------
        ratio_bounds : dict
            {'height_to_max_width': {'min': ..., 'max': ...}, ...}
        """
        hw_ratios = []
        nw_ratios = []

        for profile, meta in zip(self.profiles, self.metadata.values()):
            H = meta['height']
            r_max = meta['max_radius']

            # Neck radius (at 90% height)
            r_neck = profile.radius(0.9 * H)

            hw_ratios.append(H / (2 * r_max))
            nw_ratios.append(r_neck / r_max)

        ratios = {
            'height_to_max_width': {
                'min': np.min(hw_ratios),
                'max': np.max(hw_ratios)
            },
            'neck_to_max_width': {
                'min': np.min(nw_ratios),
                'max': np.max(nw_ratios)
            },
            'hw_ratio_mean': np.mean(hw_ratios),
            'hw_ratio_std': np.std(hw_ratios)
        }

        self.constraints['geometric_ratios'] = ratios

        print(f"\nGeometric Ratio Constraints:")
        print(f"  Height/MaxWidth: {ratios['height_to_max_width']['min']:.2f} "
              f"- {ratios['height_to_max_width']['max']:.2f}")
        print(f"  Neck/MaxWidth: {ratios['neck_to_max_width']['min']:.2f} "
              f"- {ratios['neck_to_max_width']['max']:.2f}")

        return ratios
    
    def learn_minimum_sa_threshold(self, degree: int = 2) -> Dict:
        """
        Learn minimum surface area as function of volume using polynomial regression.
        
        Hypothesis: SA_min ~ c * V^(2/3) (from sphere scaling)
        
        Parameters
        ----------
        degree : int
            Polynomial degree for regression
        
        Returns
        -------
        model_info : dict
            Contains model coefficients and R2 score
        """
        volumes = np.array([meta['volume'] for meta in self.metadata.values()])
        surface_areas = np.array([meta['surface_area'] for meta in self.metadata.values()])
        
        # Use V^(2/3) as primary feature (physical scaling)
        V_scaled = volumes ** (2/3)
        X = V_scaled.reshape(-1, 1)
        y = surface_areas
        
        # Fit model
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        r2_score = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        sphere_coeff = (36 * np.pi) ** (1 / 3)
        coeff_ratio = model.coef_[0] / sphere_coeff
        
        model_info = {
            'coefficient': model.coef_[0],
            'intercept': model.intercept_,
            'r2_score': r2_score,
            'rmse': rmse,
            'formula': f"SA_min ~ {model.coef_[0]:.4f} * V^(2/3) + {model.intercept_:.2f}",
            'interpretation': f"Empirical SA coefficient is {coeff_ratio:.3f}x sphere limit"
        }
        
        self.constraints['sa_model'] = model_info
        
        print(f"\nSurface Area Model (Lower Bound):")
        print(f"  {model_info['formula']}")
        print(f"  R2 score: {r2_score:.4f}")
        
        return model_info
    
    def cluster_bottle_types(self, n_clusters: int = 3) -> Dict:
        """
        Cluster bottles into families based on shape features.
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        
        Returns
        -------
        cluster_info : dict
            Cluster assignments and centroids
        """
        # Extract features for clustering
        features = []
        for meta in self.metadata.values():
            feat = [
                meta['height'] / meta['max_radius'],  # Aspect ratio
                meta['min_radius'] / meta['max_radius'],  # Taper ratio
                meta['sa_v_ratio']  # Efficiency
            ]
            features.append(feat)
        
        X = np.array(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_info = {
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'features': X,
            'centroids': kmeans.cluster_centers_.tolist()
        }
        
        print(f"\nBottle Type Clustering ({n_clusters} clusters):")
        for i in range(n_clusters):
            mask = labels == i
            n_in_cluster = np.sum(mask)
            names = [list(self.metadata.keys())[j] for j in range(len(labels)) if mask[j]]
            print(f"  Cluster {i}: {n_in_cluster} bottles")
            print(f"    Examples: {', '.join(names[:3])}")
        
        self.constraints['clusters'] = {
            'n_clusters': n_clusters,
            'labels': labels.tolist(),
            'features': X.tolist(),
            'centroids': kmeans.cluster_centers_.tolist()
        }

        return cluster_info
    
    def generate_optimization_constraints(self) -> List[Dict]:
        """
        Generate constraint dictionaries suitable for scipy.optimize.minimize.
        
        Returns
        -------
        constraints : list of dict
            Constraint definitions for optimizer
        """
        if 'slopes' not in self.constraints:
            raise ValueError("Must run compute_slope_constraints() first")
        
        max_slope = self.constraints['slopes']['max_slope_95th_percentile']
        
        # Slope constraint: |f'(x)| <= max_slope
        def slope_constraint(params):
            """Ensure slope doesn't exceed learned bound"""
            # This would need to be implemented with specific profile type
            # Placeholder for now
            return max_slope - 0  # Would compute actual max slope from params
        
        constraints_list = [
            {
                'type': 'ineq',
                'fun': slope_constraint,
                'name': 'max_slope'
            }
        ]
        
        return constraints_list
    
    def save_constraints(self, filepath: str = 'data/processed_profiles/learned_constraints.json'):
        """Save learned constraints to JSON file"""
        filepath = str(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.constraints, f, indent=2)
        
        print(f"\nConstraints saved to {filepath}")
    
    def summarize(self):
        """Print comprehensive summary of learned constraints"""
        print("\n" + "=" * 70)
        print("LEARNED CONSTRAINTS SUMMARY")
        print("=" * 70)
        print(f"\nData: {len(self.profiles)} bottle profiles analyzed")
        
        if self.constraints:
            print("\nConstraints Available:")
            for key in self.constraints.keys():
                print(f"  - {key}")
        else:
            print("\nNo constraints computed yet. Run analysis methods first.")
        
        print("=" * 70)


def analyze_bottle_dataset(data_dir: str = 'data/raw_profiles') -> BottleConstraintLearner:
    """
    Convenience function to run full constraint learning pipeline.
    
    Parameters
    ----------
    data_dir : str
        Directory with bottle CSV files
    
    Returns
    -------
    learner : BottleConstraintLearner
        Learner object with all constraints computed
    """
    learner = BottleConstraintLearner(data_dir)
    
    print("=" * 70)
    print("BOTTLE CONSTRAINT LEARNING")
    print("=" * 70)
    
    # Load data
    learner.load_bottle_profiles()
    
    # Compute all constraints
    learner.compute_slope_constraints()
    learner.compute_radius_constraints(n_height_sections=5)
    learner.compute_geometric_ratios()
    learner.learn_minimum_sa_threshold()
    learner.cluster_bottle_types(n_clusters=3)
    
    # Save results
    learner.save_constraints()
    
    # Summary
    learner.summarize()
    
    return learner


if __name__ == "__main__":
    print("Testing Constraints Module\n")
    
    # Run full pipeline
    learner = analyze_bottle_dataset()
    
    print("\n" + "=" * 70)
    print("Constraint learning complete!")
    print("=" * 70)
    print("\nThese constraints can now be used in optimization:")
    print("  - Max slope for manufacturability")
    print("  - Radius bounds for structural integrity")
    print("  - Geometric ratios for ergonomics")
    print("  - SA lower bounds for feasibility checks")
