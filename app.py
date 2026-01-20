from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from src.explain import intro_text, lagrange_text, math_text
from src.geometry import compute_geometry, scale_radius_to_target_volume
from src.math_utils import linspace_grid
from src.image_profile import extract_profile_from_image, sample_knots_from_profile
from src.mesh import revolve_profile_to_mesh
from src.models import (
    BulgeParams,
    CylinderParams,
    FrustumParams,
    KnotsParams,
    ShapeType,
    WaistParams,
    default_knots,
    profile_radius_function,
)
from src.optimize import cylinder_optimum_closed, dimensionless_efficiency, optimize_knots_radii
from src.viz import mesh3d_animated_figure, mesh3d_figure, profile_2d_figure


def sphere_area_lower_bound(volume_cm3: float) -> float:
    """Isoperimetric lower bound: sphere minimizes surface area for fixed volume."""
    if volume_cm3 <= 0:
        return float("nan")
    r = float((3.0 * volume_cm3 / (4.0 * np.pi)) ** (1.0 / 3.0))
    return float(4.0 * np.pi * r * r)


REAL_WORLD_PRESETS = {
    # Typical PET water bottles (very brand-dependent; good for defaults)
    330: {"height_cm": (16.0, 19.0), "diameter_cm": (5.5, 6.5)},
    500: {"height_cm": (20.0, 23.5), "diameter_cm": (6.3, 7.2)},
    1000: {"height_cm": (26.0, 30.0), "diameter_cm": (8.0, 9.0)},
    1500: {"height_cm": (30.0, 32.5), "diameter_cm": (9.0, 10.5)},
    2000: {"height_cm": (31.0, 34.0), "diameter_cm": (10.5, 11.5)},
}


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(x))))


def suggest_real_world_dimensions(volume_ml: float) -> tuple[float, float, str]:
    """Suggest (height_cm, diameter_cm) for a target volume.

    Uses nearest preset if volume is close; otherwise cube-root scaling from 500 mL.
    """
    v = float(volume_ml)
    keys = np.array(sorted(REAL_WORLD_PRESETS.keys()), dtype=float)
    nearest = float(keys[np.argmin(np.abs(keys - v))])
    rel_err = abs(nearest - v) / max(nearest, 1.0)

    if rel_err <= 0.12 and int(nearest) in REAL_WORLD_PRESETS:
        h0, h1 = REAL_WORLD_PRESETS[int(nearest)]["height_cm"]
        d0, d1 = REAL_WORLD_PRESETS[int(nearest)]["diameter_cm"]
        return float(0.5 * (h0 + h1)), float(0.5 * (d0 + d1)), f"nearest preset: {int(nearest)} mL"

    # Cube-root scaling (same-family heuristic)
    s = (v / 500.0) ** (1.0 / 3.0)
    h = 22.0 * s
    d = 6.6 * s
    return float(h), float(d), "cube-root scaling"


st.set_page_config(
    page_title="Reducing Plastic Usage Using Mathematical Optimization",
    page_icon="♻️",
    layout="wide",
)

st.title("Reducing Plastic Usage Using Mathematical Optimization")
st.caption(
    "Minimize bottle surface area (plastic) while keeping volume constant — using calculus + optimization."  # noqa: E501
)

# ---------------- Sidebar controls ----------------
with st.sidebar:
    st.header("Design Controls")

    volume_ml = st.number_input(
        "Target volume (mL)",
        min_value=50.0,
        max_value=3000.0,
        value=500.0,
        step=50.0,
    )
    target_volume_cm3 = float(volume_ml)  # 1 mL = 1 cm^3

    include_caps = st.toggle("Include top & bottom caps", value=True)

    animate_3d = st.toggle(
        "Animate 3D rotation",
        value=False,
        help="Adds a play/pause control to spin the 3D bottle around its axis.",
    )

    st.divider()
    st.subheader("Bottle family")
    shape: ShapeType = st.selectbox(
        "Shape preset",
        options=["cylinder", "frustum", "bulge", "waist", "bottle", "knots", "image"],
        index=2,
    )

    # --- Real-world sizing guide (defaults users can trust) ---
    suggested_h_cm, suggested_d_cm, sizing_source = suggest_real_world_dimensions(volume_ml)
    suggested_h_cm = _clamp(suggested_h_cm, 10.0, 40.0)
    suggested_body_r_cm = _clamp(0.5 * suggested_d_cm, 1.0, 8.0)
    suggested_shoulder_r_cm = _clamp(0.53 * suggested_d_cm, 1.0, 8.0)
    suggested_base_r_cm = _clamp(0.49 * suggested_d_cm, 1.0, 8.0)
    suggested_neck_r_cm = _clamp(0.22 * suggested_d_cm, 0.6, 5.0)

    with st.expander("Real-world sizing", expanded=False):
        st.caption("Typical sizes vary by brand; these are practical defaults.")
        st.markdown(
            "| Volume (mL) | Height (cm) | Diameter (cm) |\n"
            "|---:|---:|---:|\n"
            + "\n".join(
                f"| {v} | {REAL_WORLD_PRESETS[v]['height_cm'][0]}–{REAL_WORLD_PRESETS[v]['height_cm'][1]}"
                f" | {REAL_WORLD_PRESETS[v]['diameter_cm'][0]}–{REAL_WORLD_PRESETS[v]['diameter_cm'][1]} |"
                for v in [330, 500, 1000, 1500, 2000]
            )
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Suggested height", f"{suggested_h_cm:.1f} cm")
        c2.metric("Suggested diameter", f"{suggested_d_cm:.1f} cm")
        c3.metric("Source", sizing_source)

        if st.button("Reset sliders to suggested", use_container_width=True):
            st.session_state["height_cm"] = float(suggested_h_cm)
            # Only set bottle-specific params if the user is in the bottle preset.
            if shape == "bottle":
                st.session_state["bottle_body_radius_cm"] = float(suggested_body_r_cm)
                st.session_state["bottle_shoulder_radius_cm"] = float(suggested_shoulder_r_cm)
                st.session_state["bottle_base_radius_cm"] = float(suggested_base_r_cm)
                st.session_state["bottle_neck_radius_cm"] = float(suggested_neck_r_cm)
            st.rerun()

    height_cm = st.slider(
        "Height H (cm)",
        min_value=10.0,
        max_value=40.0,
        value=float(st.session_state.get("height_cm", 22.0)),
        step=0.5,
        key="height_cm",
    )

    auto_scale_to_volume = st.toggle(
        "Auto-scale radius to match target volume",
        value=True,
        help="Keeps your shape proportions but scales the radius so the volume matches exactly.",
    )

    st.divider()
    st.subheader("Render + accuracy")
    n_z = st.slider("Integration resolution (Nz)", 120, 900, 360, 20)
    n_theta = st.slider("3D mesh resolution (Nθ)", 48, 220, 128, 8)
    shells_n_r = st.slider("Shell-volume check (Nr)", 80, 420, 220, 20)

# Base grid
z0, z1 = 0.0, float(height_cm)
grid = linspace_grid(z0, z1, int(n_z))

# ---------------- Build baseline profile ----------------
params: object
if shape == "cylinder":
    radius_cm = st.sidebar.slider("Radius R (cm)", min_value=1.0, max_value=8.0, value=3.0, step=0.1)
    params = CylinderParams(height=height_cm, radius=radius_cm)
elif shape == "frustum":
    r0 = st.sidebar.slider("Bottom radius r0 (cm)", 1.0, 8.0, 3.5, 0.1)
    r1 = st.sidebar.slider("Top radius r1 (cm)", 1.0, 8.0, 2.5, 0.1)
    params = FrustumParams(height=height_cm, r0=r0, r1=r1)
elif shape == "bulge":
    radius_cm = st.sidebar.slider("Base radius R (cm)", 1.0, 8.0, 3.0, 0.1)
    amp = st.sidebar.slider("Bulge amount a", 0.0, 0.8, 0.25, 0.01)
    params = BulgeParams(height=height_cm, radius=radius_cm, amplitude=amp)
elif shape == "waist":
    radius_cm = st.sidebar.slider("Base radius R (cm)", 1.0, 8.0, 3.0, 0.1)
    amp = st.sidebar.slider("Waist amount a", 0.0, 0.8, 0.25, 0.01)
    params = WaistParams(height=height_cm, radius=radius_cm, amplitude=amp)
elif shape == "bottle":
    # A more bottle-like default using knots, with grouped controls.
    body_tab, neck_tab, base_tab = st.sidebar.tabs(["Body", "Neck/Cap", "Base"])

    with body_tab:
        body_radius = st.slider(
            "Body radius (cm)",
            1.0,
            8.0,
            float(st.session_state.get("bottle_body_radius_cm", 3.2)),
            0.1,
            key="bottle_body_radius_cm",
        )
        shoulder_radius = st.slider(
            "Shoulder radius (cm)",
            1.0,
            8.0,
            float(st.session_state.get("bottle_shoulder_radius_cm", 3.5)),
            0.1,
            key="bottle_shoulder_radius_cm",
        )
        shoulder_height = st.slider("Shoulder start (cm)", 4.0, 30.0, float(height_cm) * 0.68, 0.5)

    with neck_tab:
        neck_radius = st.slider(
            "Neck radius (cm)",
            0.6,
            5.0,
            float(st.session_state.get("bottle_neck_radius_cm", 1.4)),
            0.05,
            key="bottle_neck_radius_cm",
        )
        neck_height = st.slider("Neck height (cm)", 2.0, 12.0, 6.0, 0.5)
        show_top_cap = st.toggle("Show top cap in 3D", value=True)

    with base_tab:
        base_radius = st.slider(
            "Base radius (cm)",
            1.0,
            8.0,
            float(st.session_state.get("bottle_base_radius_cm", 3.0)),
            0.1,
            key="bottle_base_radius_cm",
        )
        base_lip = st.slider("Base lip height (cm)", 0.0, 4.0, 1.0, 0.25)

    # Build knot profile: base -> body -> shoulder -> neck -> top
    H = float(height_cm)
    z_knots = np.array(
        [
            0.0,
            min(base_lip, H * 0.15),
            max(min(shoulder_height, H - neck_height - 1.0), 0.0),
            max(H - neck_height, 0.0),
            H,
        ],
        dtype=float,
    )
    # Ensure strict increasing
    z_knots = np.unique(np.clip(z_knots, 0.0, H))
    if z_knots.size < 4:
        z_knots = np.linspace(0.0, H, 5, dtype=float)

    # radii aligned to z_knots (simple mapping)
    r_map = {}
    r_map[float(z_knots[0])] = float(base_radius)
    r_map[float(z_knots[-1])] = float(neck_radius)
    # Fill middle points
    for zk in z_knots[1:-1]:
        if zk <= min(base_lip, H * 0.15) + 1e-6:
            r_map[float(zk)] = float(body_radius)
        elif zk <= max(min(shoulder_height, H - neck_height - 1.0), 0.0) + 1e-6:
            r_map[float(zk)] = float(shoulder_radius)
        else:
            r_map[float(zk)] = float(neck_radius)

    r_knots = np.array([r_map[float(zk)] for zk in z_knots], dtype=float)
    params = KnotsParams(height=height_cm, knots_z=z_knots, knots_r=r_knots)

elif shape == "image":
    st.sidebar.caption("Upload a bottle silhouette and we will extract a profile.")
    up = st.sidebar.file_uploader("Upload silhouette image (png/jpg)", type=["png", "jpg", "jpeg"])
    invert = st.sidebar.toggle("Invert colors", value=False)
    threshold = st.sidebar.slider("Threshold", 0, 255, 140, 1)
    smooth_window = st.sidebar.slider("Smoothing", 1, 31, 9, 2)
    n_knots = st.sidebar.slider("Profile detail (knots)", 5, 15, 9, 1)
    base_r = st.sidebar.slider("Max radius guess (cm)", 1.0, 8.0, 3.0, 0.1)

    if up is None:
        st.sidebar.info("Upload an image to enable the custom shape.")
        params = default_knots(height=height_cm, n_knots=7, base_radius=3.0)
    else:
        extracted = extract_profile_from_image(
            up.getvalue(),
            threshold=int(threshold),
            invert=bool(invert),
            smooth_window=int(smooth_window),
        )
        z_knots, r_knots = sample_knots_from_profile(
            extracted.z_unit,
            extracted.r_unit,
            height_cm=float(height_cm),
            n_knots=int(n_knots),
            base_radius_cm=float(base_r),
        )
        params = KnotsParams(height=height_cm, knots_z=z_knots, knots_r=r_knots)

        with st.expander("Preview extracted silhouette", expanded=False):
            st.image(extracted.preview_mask, caption="Binary mask used to extract profile", clamp=True)

else:
    # knot-based profile (user-adjusted + also used for optimization)
    n_knots = st.sidebar.slider("Number of knots", 5, 11, 7, 1)
    base_r = st.sidebar.slider("Base radius (cm)", 1.0, 8.0, 3.0, 0.1)
    default = default_knots(height=height_cm, n_knots=n_knots, base_radius=base_r)

    st.sidebar.caption("Adjust knot radii (controls the silhouette)")
    knot_r = []
    for i, z_k in enumerate(default.knots_z):
        label = f"r(z={z_k:.1f}cm)"
        knot_r.append(
            st.sidebar.slider(label, min_value=0.6, max_value=8.0, value=float(default.knots_r[i]), step=0.05)
        )

    params = KnotsParams(height=height_cm, knots_z=default.knots_z, knots_r=np.array(knot_r, dtype=float))

baseline_r_of_z = profile_radius_function(shape, params)  # type: ignore[arg-type]

# Enforce volume (optional)
scale_factor = 1.0
if auto_scale_to_volume:
    baseline_r_of_z, scale_factor = scale_radius_to_target_volume(baseline_r_of_z, grid, target_volume_cm3)

baseline_geom = compute_geometry(
    baseline_r_of_z,
    grid,
    include_caps=include_caps,
    shells_n_r=int(shells_n_r),
)

# ---------------- Main layout tabs ----------------
tab_design, tab_compare, tab_theory = st.tabs(["Design", "3D Compare", "Ideas & Theory"])

z = grid.z
r_baseline = np.asarray(baseline_r_of_z(z), dtype=float)


@st.cache_data(show_spinner=False)
def mesh_from_samples(z: np.ndarray, r: np.ndarray, n_theta: int, caps: bool) -> tuple[np.ndarray, np.ndarray]:
    from src.math_utils import Grid1D

    grid_local = Grid1D(z=z)
    r_of_z = lambda zz: np.interp(np.asarray(zz, dtype=float), z, r)
    mesh = revolve_profile_to_mesh(r_of_z, grid_local, n_theta=n_theta, caps=caps, compute_normals=False)
    return mesh.vertices, mesh.faces


with tab_design:
    left, right = st.columns([1.15, 1.0], gap="large")

    with left:
        st.markdown(intro_text())
        st.markdown(math_text(include_caps=include_caps))

    with right:
        st.subheader("Key numbers")
        colA, colB, colC = st.columns(3)
        colA.metric("Volume (disk)", f"{baseline_geom.volume_cm3:.1f} cm³")
        colB.metric("Surface area", f"{baseline_geom.area_cm2:.1f} cm²")
        colC.metric("Efficiency A/V^(2/3)", f"{dimensionless_efficiency(baseline_geom.area_cm2, baseline_geom.volume_cm3):.3f}")

        st.caption(
            f"Shell-volume check: {baseline_geom.volume_shells_cm3:.1f} cm³ (should be close).\n"
            f"Auto-scale factor on radius: {scale_factor:.3f}"
        )

    st.subheader("Optimized recommendation")
    cyl_opt = cylinder_optimum_closed(target_volume_cm3)

    col1, col2, col3, col4, col5 = st.columns([1.1, 1.1, 1.1, 1.2, 1.3])
    col1.metric("Optimal cylinder R", f"{cyl_opt.radius_cm:.2f} cm")
    col2.metric("Optimal cylinder H", f"{cyl_opt.height_cm:.2f} cm")
    col3.metric("Optimal cylinder area", f"{cyl_opt.area_cm2:.1f} cm²")

    savings_pct = 100.0 * (1.0 - cyl_opt.area_cm2 / baseline_geom.area_cm2) if baseline_geom.area_cm2 > 0 else 0.0
    col4.metric("Savings vs baseline", f"{savings_pct:+.1f}%")

    if include_caps:
        sphere_area = sphere_area_lower_bound(target_volume_cm3)
        ratio = baseline_geom.area_cm2 / sphere_area if sphere_area > 0 else float("nan")
        col5.metric("Gap vs sphere bound", f"{ratio:.3f}×")
    else:
        col5.metric("Gap vs sphere bound", "N/A")

    with st.expander("Explain the cylinder result (Lagrange multipliers)", expanded=False):
        st.markdown(lagrange_text())

    st.subheader("See the shape change (2D + 3D)")
    c2d, c3d = st.columns([1.05, 1.25], gap="large")

    with c2d:
        st.plotly_chart(profile_2d_figure(z, r_baseline, r_other=None), width="stretch")
        st.caption("This is the bottle cross-section. Rotating it makes the 3D bottle.")

    with c3d:
        # For the bottle preset we may want to show caps differently (visual) while keeping math toggle separate.
        caps_for_mesh = include_caps
        if shape == "bottle":
            try:
                caps_for_mesh = bool(show_top_cap)
            except Exception:
                caps_for_mesh = include_caps

        v_base, f_base = mesh_from_samples(z, r_baseline, int(n_theta), caps_for_mesh)
        fig3d = (
            mesh3d_animated_figure(v_base, f_base, title="Bottle (3D model)")
            if animate_3d
            else mesh3d_figure(v_base, f_base, title="Bottle (3D model)")
        )
        st.plotly_chart(fig3d, width="stretch")
        st.caption("Drag to rotate • Scroll to zoom")


with tab_compare:
    st.subheader("Compare 3D shapes")

    with st.sidebar:
        st.divider()
        st.header("Optimization")
        do_opt = st.toggle("Run numeric optimization (flexible profile)", value=True)
        r_min = st.slider("Min radius (cm)", 0.6, 4.0, 1.2, 0.05)
        r_max = st.slider("Max radius (cm)", 2.0, 10.0, 6.0, 0.1)
        smooth_w = st.slider("Smoothness", 0.0, 0.2, 0.03, 0.005)
        maxiter = st.slider("Max iterations", 50, 600, 200, 25)

    opt_col_left, opt_col_right = st.columns(2, gap="large")

    v_base, f_base = mesh_from_samples(z, r_baseline, int(n_theta), include_caps)
    with opt_col_left:
        st.markdown("**Baseline (your chosen shape)**")
        fig_left = mesh3d_animated_figure(v_base, f_base, title="Baseline bottle (3D)") if animate_3d else mesh3d_figure(v_base, f_base, title="Baseline bottle (3D)")
        st.plotly_chart(fig_left, width="stretch")

    opt_result = None
    r_opt_profile = None

    if do_opt:
        if isinstance(params, KnotsParams):
            z_knots = params.knots_z
            r_knots0 = params.knots_r
        else:
            default = default_knots(height=height_cm, n_knots=7, base_radius=3.0)
            z_knots = default.knots_z
            r_knots0 = default.knots_r

        @st.cache_data(show_spinner=True)
        def run_opt(
            z_knots: np.ndarray,
            r_knots0: np.ndarray,
            target_volume: float,
            include_caps: bool,
            z_grid: np.ndarray,
            r_min: float,
            r_max: float,
            smooth_w: float,
            maxiter: int,
        ):
            from src.math_utils import Grid1D

            grid_local = Grid1D(z=z_grid)
            return optimize_knots_radii(
                knots_z=z_knots,
                initial_knots_r=r_knots0,
                target_volume_cm3=target_volume,
                grid=grid_local,
                include_caps=include_caps,
                r_min=r_min,
                r_max=r_max,
                smoothness_weight=smooth_w,
                maxiter=maxiter,
            )

        opt_result = run_opt(
            np.asarray(z_knots, dtype=float),
            np.asarray(r_knots0, dtype=float),
            float(target_volume_cm3),
            bool(include_caps),
            np.asarray(z, dtype=float),
            float(r_min),
            float(r_max),
            float(smooth_w),
            int(maxiter),
        )
        r_opt_profile = lambda zz: np.interp(np.asarray(zz, dtype=float), z_knots, opt_result.knots_r)

    with opt_col_right:
        st.markdown("**Optimized (minimum surface area at same volume)**")
        if (not do_opt) or opt_result is None or r_opt_profile is None:
            st.info("Turn on numeric optimization in the sidebar to generate the optimized 3D bottle.")
        else:
            st.caption(f"Optimization status: {'✅ success' if opt_result.success else '⚠️ not converged'} — {opt_result.message}")
            geom_opt = compute_geometry(r_opt_profile, grid, include_caps=include_caps, shells_n_r=int(shells_n_r))
            savings_opt = 100.0 * (1.0 - geom_opt.area_cm2 / baseline_geom.area_cm2) if baseline_geom.area_cm2 > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Optimized area", f"{geom_opt.area_cm2:.1f} cm²")
            c2.metric("Savings vs baseline", f"{savings_opt:+.1f}%")
            c3.metric("Volume error", f"{(geom_opt.volume_cm3 - target_volume_cm3):+.3f} cm³")

            r_opt = np.asarray(r_opt_profile(z), dtype=float)
            v_opt, f_opt = mesh_from_samples(z, r_opt, int(n_theta), include_caps)
            fig_right = mesh3d_animated_figure(v_opt, f_opt, title="Optimized bottle (3D)") if animate_3d else mesh3d_figure(v_opt, f_opt, title="Optimized bottle (3D)")
            st.plotly_chart(fig_right, width="stretch")

    st.subheader("2D profile comparison")
    r_other = None
    if r_opt_profile is not None and do_opt and opt_result is not None:
        r_other = np.asarray(r_opt_profile(z), dtype=float)
    st.plotly_chart(profile_2d_figure(z, r_baseline, r_other=r_other), width="stretch")


with tab_theory:
    st.subheader("Visual math: constraints, bounds, and tradeoffs")

    if not include_caps:
        st.info(
            "These visualizations assume a **closed shape** (caps included). "
            "Turn on 'Include top & bottom caps' to see meaningful global bounds and the cylinder optimum."
        )
    else:
        sphere_area = sphere_area_lower_bound(target_volume_cm3)
        ratio = baseline_geom.area_cm2 / sphere_area if sphere_area > 0 else float("nan")

        m1, m2, m3 = st.columns(3)
        m1.metric("Sphere lower bound (same volume)", f"{sphere_area:.1f} cm²")
        m2.metric("Baseline / sphere", f"{ratio:.3f}×")
        m3.metric("Interpretation", "Closer to 1× = less plastic")

        st.markdown("### 1) ‘Map’ view: surface area across radius & height")
        st.caption("Color shows surface area; the white curve is the fixed-volume constraint $V=V_{target}$. The red point is the optimal closed cylinder ($H=2R$).")

        R = np.linspace(0.8, 8.0, 150)
        H = np.linspace(8.0, 40.0, 150)
        RR, HH = np.meshgrid(R, H)
        V = np.pi * RR * RR * HH
        A = 2.0 * np.pi * RR * HH + 2.0 * np.pi * RR * RR

        fig_map = go.Figure()
        fig_map.add_trace(
            go.Heatmap(
                x=R,
                y=H,
                z=A,
                colorscale="Viridis",
                colorbar=dict(title="Area (cm²)"),
                hovertemplate="R=%{x:.2f} cm<br>H=%{y:.2f} cm<br>Area=%{z:.1f} cm²<extra></extra>",
            )
        )
        fig_map.add_trace(
            go.Contour(
                x=R,
                y=H,
                z=V,
                contours=dict(showlines=True, coloring="none"),
                line=dict(color="rgba(255,255,255,0.85)", width=2),
                showscale=False,
                hoverinfo="skip",
                contours_start=target_volume_cm3,
                contours_end=target_volume_cm3,
                contours_size=1,
            )
        )
        cyl_opt_local = cylinder_optimum_closed(target_volume_cm3)
        fig_map.add_trace(
            go.Scatter(
                x=[cyl_opt_local.radius_cm],
                y=[cyl_opt_local.height_cm],
                mode="markers",
                marker=dict(size=12, color="#E63946", line=dict(color="white", width=1)),
                name="Optimal cylinder",
                hovertemplate="Optimal cylinder<br>R=%{x:.2f} cm<br>H=%{y:.2f} cm<extra></extra>",
            )
        )
        fig_map.update_layout(
            height=520,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Radius R (cm)",
            yaxis_title="Height H (cm)",
        )
        st.plotly_chart(fig_map, width="stretch")

        st.markdown("### 2) Sensitivity: how the best possible area scales with volume")
        st.caption("We compare the optimal closed cylinder vs the sphere lower bound across nearby volumes.")
        v_grid = np.linspace(0.5 * target_volume_cm3, 1.5 * target_volume_cm3, 120)
        a_sphere = np.array([sphere_area_lower_bound(vv) for vv in v_grid], dtype=float)
        a_cyl = np.array([cylinder_optimum_closed(vv).area_cm2 for vv in v_grid], dtype=float)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=v_grid, y=a_sphere, mode="lines", name="Sphere bound", line=dict(width=3)))
        fig_sens.add_trace(go.Scatter(x=v_grid, y=a_cyl, mode="lines", name="Optimal closed cylinder", line=dict(width=3, dash="dash")))
        fig_sens.add_vline(x=target_volume_cm3, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.4)")
        fig_sens.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Volume (cm³)",
            yaxis_title="Surface area (cm²)",
        )
        st.plotly_chart(fig_sens, width="stretch")

        st.markdown("### 3) Tradeoff curve (fixed volume): tall vs wide")
        st.caption("Along the fixed-volume curve, changing height forces radius to change. The curve below shows how surface area changes.")
        H_curve = np.linspace(10.0, 40.0, 180)
        R_curve = np.sqrt(target_volume_cm3 / (np.pi * H_curve))
        A_curve = 2.0 * np.pi * R_curve * H_curve + 2.0 * np.pi * R_curve * R_curve
        fig_trade = go.Figure()
        fig_trade.add_trace(go.Scatter(x=H_curve, y=A_curve, mode="lines", name="Area on V=const", line=dict(width=3)))
        fig_trade.add_trace(
            go.Scatter(
                x=[cyl_opt_local.height_cm],
                y=[cyl_opt_local.area_cm2],
                mode="markers",
                marker=dict(size=12, color="#E63946", line=dict(color="white", width=1)),
                name="Optimum",
            )
        )
        fig_trade.update_layout(
            height=380,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis_title="Height H (cm)",
            yaxis_title="Surface area (cm²)",
        )
        st.plotly_chart(fig_trade, width="stretch")

        st.caption(
            "Want something even more ‘map-like’? Next we can add a true multi-objective slider (area vs max height/diameter) "
            "and show a Pareto scatter plot for your flexible bottle profile."
        )
