from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

def profile_2d_figure(
    z: np.ndarray,
    r_baseline: np.ndarray,
    r_other: np.ndarray | None = None,
    other_name: str = "Optimized",
) -> go.Figure:
    """2D silhouette of the bottle (cross-section).

    Plots the outline x = ±r(z) against height z, and fills the shape so it looks
    like a bottle instead of two mirrored lines.
    """
    z = np.asarray(z, dtype=float)
    r_baseline = np.asarray(r_baseline, dtype=float)

    fig = go.Figure()

    # Baseline filled silhouette
    xb = np.concatenate([r_baseline, -r_baseline[::-1]])
    zb = np.concatenate([z, z[::-1]])
    fig.add_trace(
        go.Scatter(
            x=xb,
            y=zb,
            mode="lines",
            name="Baseline",
            fill="toself",
            fillcolor="rgba(46,134,171,0.18)",
            line=dict(width=3, color="#2E86AB"),
        )
    )

    # Optional overlay silhouette
    if r_other is not None:
        r_other = np.asarray(r_other, dtype=float)
        xo = np.concatenate([r_other, -r_other[::-1]])
        zo = np.concatenate([z, z[::-1]])
        fig.add_trace(
            go.Scatter(
                x=xo,
                y=zo,
                mode="lines",
                name=other_name,
                fill="toself",
                fillcolor="rgba(246,137,52,0.12)",
                line=dict(width=3, dash="dash", color="#F68934"),
            )
        )

    # Centerline
    fig.add_trace(
        go.Scatter(
            x=np.zeros_like(z),
            y=z,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.18)", width=1),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=dict(
            text="Bottle Profile (2D cross-section)",
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
            pad=dict(b=8),
        ),
        xaxis_title="Radius (cm)",
        yaxis_title="Height z (cm)",
        height=460,
        margin=dict(l=40, r=20, t=80, b=45),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0),
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(zeroline=False)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def mesh3d_figure(vertices: np.ndarray, faces: np.ndarray, title: str) -> go.Figure:
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                opacity=0.95,
                color="#2E86AB",
                flatshading=False,
                lighting=dict(ambient=0.35, diffuse=0.9, specular=0.2, roughness=0.9),
                lightposition=dict(x=100, y=100, z=80),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(title="x (cm)", showgrid=False, zeroline=False),
            yaxis=dict(title="y (cm)", showgrid=False, zeroline=False),
            zaxis=dict(title="z (cm)", showgrid=False, zeroline=False),
            aspectmode="data",
        ),
        showlegend=False,
    )

    return fig


def mesh3d_animated_figure(
    vertices: np.ndarray,
    faces: np.ndarray,
    title: str,
    n_frames: int = 72,
    orbit_radius: float = 2.2,
) -> go.Figure:
    """3D mesh with a built-in camera rotation animation (play/pause)."""
    fig = mesh3d_figure(vertices, faces, title=title)

    frames = []
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_frames), endpoint=False)
    for a in angles:
        eye = dict(x=float(orbit_radius * np.cos(a)), y=float(orbit_radius * np.sin(a)), z=1.0)
        frames.append(go.Frame(layout=dict(scene_camera=dict(eye=eye))))

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.0,
                y=1.12,
                xanchor="left",
                yanchor="top",
                direction="right",
                buttons=[
                    dict(
                        label="▶ Play rotation",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=35, redraw=False),
                                transition=dict(duration=0),
                                fromcurrent=True,
                                mode="immediate",
                                loop=True,
                            ),
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ]
    )
    return fig
