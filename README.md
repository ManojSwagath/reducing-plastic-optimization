# Reducing Plastic Usage Using Mathematical Optimization

Interactive Streamlit web app demonstrating how **mathematical optimization** can reduce plastic usage by minimizing **surface area** (plastic) while maintaining a **fixed volume** (capacity) for water-bottle-like shapes.

## What this app shows
- Bottle shapes modeled as **surfaces of revolution** from a radius profile \(r(z)\).
- **Volume** computed two ways:
  - Disk method: \(V = \pi \int_0^H r(z)^2\,dz\)
  - Cylindrical shells (numerical check/intuition): \(V = \int_0^{r_{max}} 2\pi\rho\,h(\rho)\,d\rho\)
- **Surface area** (plastic usage proxy):
  - Lateral area: \(A = 2\pi \int_0^H r(z)\sqrt{1+(r'(z))^2}\,dz\)
  - Optional caps can be toggled in the UI.
- Optimization:
  - **Analytic cylinder optimum** via Lagrange multipliers (closed cylinder): \(H=2R\)
  - **Numeric optimization** for a flexible profile (knot radii) with \(V=V_{target}\)

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app.
3. Set the entrypoint to `app.py`.

## Resume-ready talking points
- Built an interactive optimization demo with calculus-based surface/volume computation.
- Implemented constrained optimization (fixed volume) and interactive 2D/3D visualization.
- Demonstrated real-world sustainability impact by reducing surface area (plastic) for fixed capacity.
