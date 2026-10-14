"""
Plastic Bottle Optimizer — Streamlit Web App
============================================
Combines mathematical shape optimization (SLSQP) with a scikit-learn
Random Forest model to predict bottle mass and estimate plastic savings.

Run:  streamlit run app.py
"""

import sys
import os

# Make sure src/ is importable whether running from repo root or elsewhere
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

from geometry import BottleProfile, surface_area, volume
from optimization import optimize_bottle

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plastic Bottle Optimizer",
    page_icon="🍶",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helper: build synthetic training data for the RF mass model
# (labelled clearly as synthetic — no real data available at training time)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_training_data(n_samples: int = 500):
    """
    Generate synthetic bottle dataset for RF training.

    Features : height_cm, max_radius_cm, capacity_ml, surface_area_cm2
    Target   : mass_g  (wall_thickness × SA × PET_density, with noise)

    NOTE: This is SYNTHETIC data generated from physical scaling laws.
    It is used only because no real measured mass dataset is bundled
    with this repository.  Replace with real data by swapping the
    DataFrame source below.
    """
    rng = np.random.default_rng(42)

    heights      = rng.uniform(10, 30, n_samples)          # cm
    max_radii    = rng.uniform(1.5, 6.0, n_samples)        # cm
    # wall thickness correlates with SA (thinner for large bottles)
    wall_mm      = rng.uniform(0.2, 0.6, n_samples)        # mm

    # Approximate SA for a tapered cylinder
    neck_radii   = max_radii * rng.uniform(0.4, 0.8, n_samples)
    sa_approx    = (
        np.pi * (max_radii + neck_radii)
        * np.sqrt(heights**2 + (max_radii - neck_radii)**2)
        + np.pi * max_radii**2
    )                                                       # cm²

    # Volume ≈ π * avg_r² * h
    avg_radii    = (max_radii + neck_radii) / 2
    capacity_ml  = np.pi * avg_radii**2 * heights           # cm³ = mL

    # Mass: wall_thickness × SA × PET density (1.38 g/cm³) + noise
    PET_density  = 1.38                                     # g/cm³
    wall_cm      = wall_mm / 10
    mass_g       = wall_cm * sa_approx * PET_density
    mass_g      += rng.normal(0, mass_g * 0.05)            # 5 % noise

    df = pd.DataFrame({
        "height_cm":       heights,
        "max_radius_cm":   max_radii,
        "capacity_ml":     capacity_ml,
        "surface_area_cm2": sa_approx,
        "mass_g":          mass_g,
    })
    return df


@st.cache_resource(show_spinner=False)
def train_rf_model():
    """Train Random Forest on synthetic bottle data and return (model, metrics)."""
    df = build_training_data()
    features = ["height_cm", "max_radius_cm", "capacity_ml", "surface_area_cm2"]

    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["mass_g"], test_size=0.2, random_state=42
    )
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    metrics = {
        "r2":  r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }
    return rf, metrics, features


# ---------------------------------------------------------------------------
# Helper: build a BottleProfile from simple scalar inputs
# ---------------------------------------------------------------------------

def make_profile_from_inputs(
    height: float,
    max_radius: float,
    neck_radius: float,
    name: str = "Bottle",
) -> BottleProfile:
    """Create a B-spline BottleProfile from three key dimensions."""
    # 6 control radii: base → widest point → shoulder → neck
    control_radii = [
        max_radius * 0.90,   # base
        max_radius * 1.00,   # widest (lower body)
        max_radius * 0.95,   # mid body
        max_radius * 0.80,   # upper shoulder
        neck_radius  * 1.10, # throat
        neck_radius,         # top (neck)
    ]
    return BottleProfile.from_bspline(control_radii, height=height, name=name)


# ---------------------------------------------------------------------------
# Helper: Plotly 3D surface of revolution
# ---------------------------------------------------------------------------

def plotly_bottle_3d(
    profile: BottleProfile,
    color: str = "#2196F3",
    name: str = "Bottle",
    opacity: float = 0.75,
    n_theta: int = 48,
    n_x: int = 80,
) -> go.Surface:
    """Return a Plotly Surface trace for a bottle profile."""
    x_min, x_max = profile.height_range
    x_vals   = np.linspace(x_min, x_max, n_x)
    theta    = np.linspace(0, 2 * np.pi, n_theta)
    X, Theta = np.meshgrid(x_vals, theta)
    R        = profile.radius(X)
    Y        = R * np.cos(Theta)
    Z        = R * np.sin(Theta)

    return go.Surface(
        x=X, y=Y, z=Z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        opacity=opacity,
        name=name,
        hoverinfo="skip",
    )


def build_3d_figure(profiles: list, names: list, colors: list) -> go.Figure:
    """Combine one or two bottle traces into a single Plotly figure."""
    fig = go.Figure()
    for prof, nm, col in zip(profiles, names, colors):
        fig.add_trace(plotly_bottle_3d(prof, color=col, name=nm))

    fig.update_layout(
        scene=dict(
            xaxis_title="Height (cm)",
            yaxis_title="X (cm)",
            zaxis_title="Y (cm)",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        height=500,
        legend=dict(x=0.01, y=0.99),
        paper_bgcolor="#f8f9fa",
    )
    return fig


# ---------------------------------------------------------------------------
# Predict mass with the RF model
# ---------------------------------------------------------------------------

def predict_mass(
    rf_model,
    feature_names: list,
    height: float,
    max_radius: float,
    capacity: float,
    sa: float,
) -> float:
    row = pd.DataFrame(
        [[height, max_radius, capacity, sa]],
        columns=feature_names,
    )
    return float(rf_model.predict(row)[0])


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.title("🍶 Plastic Bottle Shape Optimizer")
    st.markdown(
        "Minimise plastic use with **mathematical optimisation** (SLSQP) "
        "and a **Random Forest** mass predictor."
    )

    # ---- sidebar ---------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Bottle Parameters")
        target_volume = st.slider(
            "Target capacity (mL)", 250, 2000, 500, step=50
        )
        st.markdown("---")
        st.subheader("Initial shape")
        init_height     = st.slider("Height (cm)",       10.0, 30.0, 20.0, 0.5)
        init_max_radius = st.slider("Max radius (cm)",    1.5,  7.0,  3.5, 0.1)
        init_neck_ratio = st.slider("Neck / max-radius", 0.30, 0.90, 0.55, 0.05)
        init_neck_radius = init_max_radius * init_neck_ratio

        st.markdown("---")
        run_opt = st.button("🚀 Run Optimisation", use_container_width=True)

        st.markdown("---")
        st.caption(
            "**Note:** The Random Forest is trained on **synthetic** data "
            "generated from physical scaling laws (PET density 1.38 g/cm³, "
            "wall thickness 0.2–0.6 mm). Replace the training dataset with "
            "real measurements for production use."
        )

    # ---- load model -------------------------------------------------------
    with st.spinner("Training Random Forest model…"):
        rf_model, rf_metrics, feature_names = train_rf_model()

    # ---- original bottle --------------------------------------------------
    orig_profile = make_profile_from_inputs(
        init_height, init_max_radius, init_neck_radius, name="Original"
    )
    orig_sa  = surface_area(orig_profile)
    orig_vol = volume(orig_profile)
    orig_mass = predict_mass(
        rf_model, feature_names,
        init_height, init_max_radius, orig_vol, orig_sa
    )

    # ---- optimisation (on demand) ----------------------------------------
    opt_result   = None
    opt_profile  = None
    opt_sa       = None
    opt_vol      = None
    opt_mass     = None

    if run_opt:
        with st.spinner("Running SLSQP shape optimisation…"):
            opt_result = optimize_bottle(
                target_volume=target_volume,
                n_control_points=6,
                method="SLSQP",
                verbose=False,
            )
        if opt_result.success:
            opt_profile = opt_result.profile
            opt_sa      = opt_result.final_surface_area
            opt_vol     = opt_result.final_volume
            # Derive max radius from optimised profile
            x_vals       = np.linspace(*opt_profile.height_range, 200)
            opt_max_r    = float(np.max(opt_profile.radius(x_vals)))
            opt_height   = float(opt_profile.height_range[1] - opt_profile.height_range[0])
            opt_mass = predict_mass(
                rf_model, feature_names,
                opt_height, opt_max_r, opt_vol, opt_sa
            )
        else:
            st.error(f"Optimisation failed: {opt_result.message}")

    # ---- layout: metrics row ---------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Original capacity",
            f"{orig_vol:.0f} mL",
            help="Volume computed by numerical integration of the profile",
        )
    with col2:
        st.metric("Original surface area", f"{orig_sa:.1f} cm²")
    with col3:
        st.metric("Predicted mass (RF)", f"{orig_mass:.1f} g",
                  help="Random Forest prediction — synthetic training data")
    with col4:
        if opt_mass is not None:
            savings_pct = (orig_mass - opt_mass) / orig_mass * 100
            st.metric(
                "Estimated plastic saving",
                f"{orig_mass - opt_mass:.1f} g",
                delta=f"{-savings_pct:.1f}%",
                delta_color="inverse",
            )
        else:
            st.metric("Estimated plastic saving", "—", help="Run optimisation first")

    st.markdown("---")

    # ---- 3D visualisation ------------------------------------------------
    vis_col, info_col = st.columns([3, 2])

    with vis_col:
        st.subheader("3D Bottle Visualisation")
        if opt_profile is not None:
            fig = build_3d_figure(
                [orig_profile, opt_profile],
                ["Original", "Optimised"],
                ["#2196F3", "#4CAF50"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = build_3d_figure(
                [orig_profile],
                ["Original"],
                ["#2196F3"],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Click **Run Optimisation** to compare original vs optimised.")

    with info_col:
        st.subheader("📊 Shape Comparison")

        rows = {
            "Metric": ["Height (cm)", "Max radius (cm)", "Capacity (mL)",
                       "Surface area (cm²)", "SA/V ratio (cm⁻¹)",
                       "Predicted mass (g)"],
            "Original": [
                f"{init_height:.1f}",
                f"{init_max_radius:.2f}",
                f"{orig_vol:.0f}",
                f"{orig_sa:.1f}",
                f"{orig_sa/orig_vol:.4f}",
                f"{orig_mass:.1f}",
            ],
        }

        if opt_profile is not None:
            x_v = np.linspace(*opt_profile.height_range, 200)
            opt_h = float(opt_profile.height_range[1] - opt_profile.height_range[0])
            opt_r = float(np.max(opt_profile.radius(x_v)))
            rows["Optimised"] = [
                f"{opt_h:.1f}",
                f"{opt_r:.2f}",
                f"{opt_vol:.0f}",
                f"{opt_sa:.1f}",
                f"{opt_sa/opt_vol:.4f}",
                f"{opt_mass:.1f}",
            ]

        st.dataframe(
            pd.DataFrame(rows).set_index("Metric"),
            use_container_width=True,
        )

        if opt_profile is not None and opt_mass is not None:
            saving_g   = orig_mass - opt_mass
            saving_pct = saving_g / orig_mass * 100
            st.success(
                f"✅ Optimised shape uses **{saving_g:.1f} g less plastic** "
                f"({saving_pct:.1f}% reduction) while meeting the "
                f"{target_volume} mL capacity target."
            )

    st.markdown("---")

    # ---- 2D profile comparison -------------------------------------------
    st.subheader("2D Profile Cross-section")

    x_orig = np.linspace(*orig_profile.height_range, 300)
    r_orig = orig_profile.radius(x_orig)

    fig2d = go.Figure()
    # Mirror both sides
    fig2d.add_trace(go.Scatter(
        x=list(r_orig) + list(-r_orig[::-1]),
        y=list(x_orig) + list(x_orig[::-1]),
        fill="toself", fillcolor="rgba(33,150,243,0.18)",
        line=dict(color="#2196F3", width=2),
        name="Original",
    ))

    if opt_profile is not None:
        x_opt = np.linspace(*opt_profile.height_range, 300)
        r_opt = opt_profile.radius(x_opt)
        fig2d.add_trace(go.Scatter(
            x=list(r_opt) + list(-r_opt[::-1]),
            y=list(x_opt) + list(x_opt[::-1]),
            fill="toself", fillcolor="rgba(76,175,80,0.18)",
            line=dict(color="#4CAF50", width=2),
            name="Optimised",
        ))

    fig2d.update_layout(
        xaxis_title="Radius (cm)",
        yaxis_title="Height (cm)",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="#f8f9fa",
        plot_bgcolor="#f8f9fa",
    )
    st.plotly_chart(fig2d, use_container_width=True)

    st.markdown("---")

    # ---- RF model info ---------------------------------------------------
    with st.expander("🤖 Random Forest Model Details", expanded=False):
        st.markdown(
            "**Training data:** Synthetic — generated from physical scaling "
            "laws (PET density 1.38 g/cm³, wall thickness 0.2–0.6 mm, "
            "500 samples). No real measured masses are included.\n\n"
            "**Features:** `height_cm`, `max_radius_cm`, `capacity_ml`, "
            "`surface_area_cm2`\n\n"
            "**Target:** `mass_g`"
        )
        m1, m2 = st.columns(2)
        m1.metric("R² (test set)", f"{rf_metrics['r2']:.4f}")
        m2.metric("MAE (test set)", f"{rf_metrics['mae']:.2f} g")

        st.markdown("**Feature importances**")
        df_imp = build_training_data()   # just to get feature list
        feat_imp = pd.Series(
            rf_model.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)
        st.bar_chart(feat_imp)

    # ---- optimisation details -------------------------------------------
    if opt_result is not None:
        with st.expander("🔧 Optimisation Details", expanded=False):
            st.markdown(
                f"**Solver:** SLSQP (Sequential Least Squares Programming)  \n"
                f"**Iterations:** {opt_result.num_iterations}  \n"
                f"**Status:** {'✅ Converged' if opt_result.success else '❌ ' + opt_result.message}  \n"
                f"**Objective (SA):** {opt_result.final_surface_area:.2f} cm²  \n"
                f"**Achieved volume:** {opt_result.final_volume:.1f} mL"
            )
            if len(opt_result.convergence_history) > 1:
                st.markdown("**Convergence history**")
                hist = opt_result.convergence_history
                fig_conv = go.Figure(go.Scatter(
                    y=hist, mode="lines",
                    line=dict(color="#7c5cd8"),
                ))
                fig_conv.update_layout(
                    xaxis_title="Evaluation #",
                    yaxis_title="Surface area (cm²)",
                    height=250,
                    margin=dict(l=0, r=0, t=10, b=0),
                    paper_bgcolor="#f8f9fa",
                    plot_bgcolor="#f8f9fa",
                )
                st.plotly_chart(fig_conv, use_container_width=True)


if __name__ == "__main__":
    main()
