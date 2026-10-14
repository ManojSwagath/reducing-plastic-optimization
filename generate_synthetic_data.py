"""
Generate synthetic bottle profile data.

Creates realistic bottle profiles with various shapes:
- Standard cylindrical
- Tapered
- Ergonomic (curved body)
- Sports bottle (narrow neck)
- Wide-mouth jar

Each profile saved as CSV with columns: height (cm), radius (cm)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Simplified interpolation (avoid scipy loading issues on some systems)
def simple_interpolate(x_control, y_control, x_dense):
    """Simple linear interpolation, then smoothed with conv"""
    y_dense = np.interp(x_dense, x_control, y_control)
    # Apply smoothing
    window = 5
    kernel = np.ones(window) / window
    y_smooth = np.convolve(y_dense, kernel, mode='same')
    return y_smooth


def add_measurement_noise(radii, noise_std=0.02):
    """Add realistic measurement noise to radii (~0.2mm std)"""
    return radii + np.random.normal(0, noise_std, size=len(radii))


def generate_standard_bottle():
    """Standard water bottle: cylindrical body, tapered neck"""
    heights = np.array([0, 2, 4, 8, 12, 16, 18, 19, 20])
    radii = np.array([3.0, 3.0, 3.2, 3.3, 3.2, 3.0, 2.5, 2.2, 2.0])
    
    # Interpolate to get smooth profile
    h_smooth = np.linspace(0, 20, 100)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def generate_cylindrical_bottle():
    """Simple cylindrical bottle"""
    heights = np.linspace(0, 15, 100)
    radii = np.ones(100) * 2.8
    radii = add_measurement_noise(radii, noise_std=0.015)
    
    return pd.DataFrame({'height_cm': heights, 'radius_cm': radii})


def generate_tapered_bottle():
    """Linearly tapered from base to top"""
    heights = np.linspace(0, 18, 100)
    radii = 4.0 - (4.0 - 2.0) * (heights / 18)  # Taper from 4cm to 2cm
    radii = add_measurement_noise(radii)
    
    return pd.DataFrame({'height_cm': heights, 'radius_cm': radii})


def generate_ergonomic_bottle():
    """Bottle with grip curve (indentation mid-body)"""
    heights = np.array([0, 3, 6, 9, 12, 15, 17, 19, 22])
    radii = np.array([3.5, 3.8, 4.0, 3.5, 3.6, 3.8, 3.2, 2.5, 2.0])  # Grip at h=9cm
    
    h_smooth = np.linspace(0, 22, 110)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def generate_sports_bottle():
    """Sports bottle: wide body, very narrow neck"""
    heights = np.array([0, 2, 5, 10, 15, 18, 20, 21, 22, 23])
    radii = np.array([3.5, 3.8, 4.2, 4.5, 4.3, 3.5, 2.5, 2.0, 1.8, 1.5])
    
    h_smooth = np.linspace(0, 23, 120)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def generate_wide_mouth_bottle():
    """Wide-mouth bottle/jar"""
    heights = np.linspace(0, 12, 100)
    radii = 4.5 - 0.3 * np.sin(np.pi * heights / 12)  # Slight bulge
    radii = add_measurement_noise(radii, noise_std=0.025)
    
    return pd.DataFrame({'height_cm': heights, 'radius_cm': radii})


def generate_small_bottle():
    """Small bottle (250 mL)"""
    heights = np.array([0, 2, 4, 6, 8, 10, 11, 12])
    radii = np.array([2.5, 2.8, 3.0, 2.9, 2.7, 2.5, 2.0, 1.8])
    
    h_smooth = np.linspace(0, 12, 80)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def generate_large_bottle():
    """Large bottle (1 liter)"""
    heights = np.array([0, 3, 6, 10, 15, 20, 23, 25, 26])
    radii = np.array([4.5, 5.0, 5.3, 5.5, 5.4, 5.0, 4.0, 3.0, 2.5])
    
    h_smooth = np.linspace(0, 26, 130)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def generate_contoured_bottle():
    """Bottle with multiple contours"""
    heights = np.array([0, 2, 5, 8, 11, 14, 17, 19, 21])
    radii = np.array([3.2, 3.5, 3.8, 3.4, 3.7, 4.0, 3.5, 2.8, 2.2])
    
    h_smooth = np.linspace(0, 21, 105)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def generate_classic_soda_bottle():
    """Classic soda bottle shape"""
    heights = np.array([0, 3, 6, 9, 12, 16, 19, 21, 23, 24])
    radii = np.array([4.5, 5.0, 4.8, 4.5, 4.3, 3.8, 3.0, 2.3, 1.8, 1.5])
    
    h_smooth = np.linspace(0, 24, 120)
    r_smooth = simple_interpolate(heights, radii, h_smooth)
    r_smooth = add_measurement_noise(r_smooth)
    
    return pd.DataFrame({'height_cm': h_smooth, 'radius_cm': r_smooth})


def main():
    """Generate all synthetic bottle profiles"""
    
    # Create output directory
    output_dir = 'data/raw_profiles'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate bottles
    bottles = {
        'bottle_01_standard.csv': generate_standard_bottle(),
        'bottle_02_cylindrical.csv': generate_cylindrical_bottle(),
        'bottle_03_tapered.csv': generate_tapered_bottle(),
        'bottle_04_ergonomic.csv': generate_ergonomic_bottle(),
        'bottle_05_sports.csv': generate_sports_bottle(),
        'bottle_06_wide_mouth.csv': generate_wide_mouth_bottle(),
        'bottle_07_small.csv': generate_small_bottle(),
        'bottle_08_large.csv': generate_large_bottle(),
        'bottle_09_contoured.csv': generate_contoured_bottle(),
        'bottle_10_soda.csv': generate_classic_soda_bottle(),
    }
    
    # Save to CSV
    print("Generating synthetic bottle profiles...")
    print("=" * 60)
    
    for filename, df in bottles.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        # Calculate approximate volume for metadata
        h = df['height_cm'].values
        r = df['radius_cm'].values
        vol_approx = np.pi * np.trapezoid(r**2, h)  # cm³ = mL
        
        print(f"✓ {filename:30s} - {len(df)} points, ~{vol_approx:.0f} mL")
    
    print("=" * 60)
    print(f"Generated {len(bottles)} bottle profiles in {output_dir}/")
    
    # Create metadata file
    metadata = {
        'bottle_01_standard.csv': {
            'description': 'Standard water bottle with tapered neck',
            'type': 'standard',
            'nominal_volume_ml': 500
        },
        'bottle_02_cylindrical.csv': {
            'description': 'Simple cylindrical bottle',
            'type': 'cylindrical',
            'nominal_volume_ml': 350
        },
        'bottle_03_tapered.csv': {
            'description': 'Linearly tapered from base to top',
            'type': 'tapered',
            'nominal_volume_ml': 550
        },
        'bottle_04_ergonomic.csv': {
            'description': 'Ergonomic bottle with grip indentation',
            'type': 'ergonomic',
            'nominal_volume_ml': 650
        },
        'bottle_05_sports.csv': {
            'description': 'Sports bottle with wide body and narrow neck',
            'type': 'sports',
            'nominal_volume_ml': 750
        },
        'bottle_06_wide_mouth.csv': {
            'description': 'Wide-mouth bottle/jar',
            'type': 'wide_mouth',
            'nominal_volume_ml': 700
        },
        'bottle_07_small.csv': {
            'description': 'Small bottle for single serving',
            'type': 'small',
            'nominal_volume_ml': 250
        },
        'bottle_08_large.csv': {
            'description': 'Large 1-liter bottle',
            'type': 'large',
            'nominal_volume_ml': 1000
        },
        'bottle_09_contoured.csv': {
            'description': 'Bottle with multiple body contours',
            'type': 'contoured',
            'nominal_volume_ml': 600
        },
        'bottle_10_soda.csv': {
            'description': 'Classic soda bottle shape',
            'type': 'soda',
            'nominal_volume_ml': 600
        },
    }
    
    # Save metadata as JSON
    import json
    metadata_path = 'data/processed_profiles/metadata.json'
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Metadata saved to {metadata_path}")
    print("\nReady for analysis in notebooks!")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
