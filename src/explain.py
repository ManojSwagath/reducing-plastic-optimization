from __future__ import annotations


def intro_text() -> str:
    return (
        "### Why surface area matters\n"
        "If two bottles hold the **same volume of water**, the one with **smaller surface area** uses **less plastic** "
        "(assuming the same thickness). This app shows how calculus and optimization can recommend shapes that reduce plastic.\n"
    )


def math_text(include_caps: bool) -> str:
    caps_note = "including top and bottom caps" if include_caps else "(lateral area only; caps not counted)"
    return (
        "### The math model (in simple terms)\n"
        r"We describe the bottle by its **radius profile** $r(z)$ along the height $z\in[0,H]$." + "\n\n"
        "**Volume** (disk method):\n"
        r"$${V = \pi \int_0^H r(z)^2\,dz}$$" + "\n"
        "**Volume** (cylindrical shells intuition):\n"
        r"$${V = \int_0^{r_{max}} 2\pi\rho\,h(\rho)\,d\rho}$$" + "\n"
        r"where $h(\rho)$ is how much of the height has radius at least $\rho$." + "\n\n"
        f"**Surface area** {caps_note}:\n"
        r"$${A = 2\pi\int_0^H r(z)\sqrt{1+(r'(z))^2}\,dz + A_{caps}}$$" + "\n"
    )


def lagrange_text() -> str:
    return (
        "### Lagrange multipliers (cylinder result)\n"
        "For a **closed cylinder** with radius $R$ and height $H$:\n"
        r"$${A = 2\pi R H + 2\pi R^2, \quad V = \pi R^2 H}$$" + "\n"
        "Minimizing $A$ subject to fixed $V$ gives the famous optimum:\n"
        r"$${H = 2R}$$" + "\n"
        "Meaning: the height equals the diameter for the least plastic (for a cylinder).\n"
    )
