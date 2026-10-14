# Implementation Progress

## Date
- 2026-03-17

## Status
- Phase 1 started and completed: reproducibility blocker fixes for notebook-to-module compatibility.

## Changes Implemented

### 1. `src/geometry.py`
- Added backward-compatible `BottleProfile` constructor support:
  - `radius_func` + `height` aliases are now accepted.
- Added `height` property derived from `height_range` for notebook compatibility.

### 2. `src/optimization.py`
- Added optional `bounds_radii` parameter to `optimize_bottle(...)`.
- Integrated `bounds_radii` into default B-spline bounds when explicit `bounds` are not provided.

### 3. `src/constraints.py`
- Added dictionary-like compatibility for notebooks:
  - `__contains__`, `__getitem__`, `keys`, `items`.
  - Normalized view keys now include:
    - `slope_constraints`
    - `radius_constraints`
    - `sa_threshold`
    - `bottle_clusters`
- `load_bottle_profiles(...)` now optionally accepts a data directory argument and safely resets internal data on repeated calls.
- `compute_radius_constraints(...)` now returns a list of section dictionaries expected by notebooks.
- `compute_geometric_ratios(...)` output format aligned with notebook access patterns (`min`/`max` nested keys).
- `learn_minimum_sa_threshold(...)` now includes:
  - `rmse`
  - `interpretation`
- `cluster_bottle_types(...)` now returns features as an array for notebook indexing while storing JSON-safe lists in saved constraints.
- `save_constraints(...)` now accepts `Path` inputs safely.
- Replaced non-ASCII status symbols in output strings with ASCII for Windows console compatibility.

### 4. `data/processed_profiles/metadata.json`
- Added `filename` field for each bottle entry to match notebook loader expectations.

## Validation Performed
- Python environment configured and used for smoke test.
- Verified no syntax/lint errors in:
  - `src/geometry.py`
  - `src/optimization.py`
  - `src/constraints.py`
- Smoke-tested compatibility paths successfully:
  - `BottleProfile(radius_func=..., height=...)`
  - `optimize_bottle(..., bounds_radii=...)`
  - `analyze_bottle_dataset(...)` with notebook-style dictionary key access
  - radius sections and cluster feature shape outputs

## Notes
- Numerical integration warnings may still appear for some rough profiles during optimization; these are warnings (not crashes) and can be addressed in a later numerical-stability pass.

## Next Implementation Step
- Phase 2: execute notebook-by-notebook (`01` -> `05`) and patch any remaining runtime-cell issues, then update README/theory claims only where outputs differ from documented observations.

## Update: Phase 2 (Notebook 01)
- Executed `notebooks/01_geometry_modeling.ipynb` end-to-end.
- Identified and fixed a Matplotlib shared-axis aspect conflict in plotting utility.
  - Updated `src/visualization.py` to use `adjustable='box'` for equal-aspect profile plots.
- Re-ran notebook after kernel restart; all code cells completed successfully.

## Deferred Final Task File
- Added `TODO_LAST.md` to track deferred finalization scope for:
  - Dockerization with pinned Python/dependencies
  - single-command Windows batch launcher
  - documentation updates for one-command cross-device execution

## Paper Draft Milestone
- Created first manuscript draft from Notebook 01 results:
  - `papers/01_geometry_modeling_paper.md`
- Draft includes abstract, formulation, methods, results, limitations, and next-step framing for optimization-stage paper.
