# Thermal Simulation Engine

## Overview
A numerical solver for the **2D Transient Heat Equation** using the explicit Finite Difference Method (FDM). Simulates how heat diffuses across a steel plate over time given fixed boundary conditions, and renders the thermal gradient evolution as contour maps.

**Tech Stack:** Python · NumPy · Matplotlib

---

## Physics Background

The simulation solves the 2D heat equation:

```
∂u/∂t = α (∂²u/∂x² + ∂²u/∂y²)
```

Where:
- `u(x, y, t)`, temperature at position (x, y) at time t
- `α = 0.0001 m²/s`, thermal diffusivity of steel

### Numerical Method
The spatial derivatives (Laplacian) are approximated using the **central difference scheme**:

```
∂²u/∂x² ≈ (u[i+1,j] - 2u[i,j] + u[i-1,j]) / dx²
```

Time integration uses the **explicit Forward Euler method**:

```
u_next[i,j] = u[i,j] + α·dt·(∂²u/∂x² + ∂²u/∂y²)
```

### Stability
The time step `dt` is constrained by the **Von Neumann stability criterion** for 2D explicit methods:

```
dt ≤ dx² / (4α)
```

A safety factor of 0.9 is applied to prevent numerical blow-up:
```python
dt = (dx**2 / (4 * alpha)) * 0.9
```

---

## Configuration

| Parameter | Value | Description |
|---|---|---|
| Grid resolution | 50 × 50 | Spatial discretisation |
| Physical plate size | 1.0 m × 1.0 m | Domain size |
| Thermal diffusivity (α) | 0.0001 m²/s | Steel |
| Time steps | 3,000 | Simulation duration |
| Top boundary | 100°C | Boiling water (Dirichlet) |
| Bottom boundary | 0°C | Ice (Dirichlet) |
| Left/Right boundaries | 20°C | Room temperature (insulated) |

---

## Output
The simulation captures thermal snapshots at iterations 0, 100, 1000, and 2999, rendered as colour-mapped heatmaps using the `inferno` colormap.

The progression shows heat diffusing from the top boundary (100°C) toward the cooler bottom boundary (0°C) until a near-steady-state gradient is reached.

---

## Usage

**1. Install dependencies:**
```bash
pip install numpy matplotlib
```

**2. Run the simulation:**
```bash
python thermal_simulation_engine.py
```

A 2×2 grid of thermal contour maps will render showing the plate's temperature distribution at four points in time.

---

## Key Concepts Demonstrated
- Finite Difference Method for solving PDEs
- Von Neumann stability analysis for explicit time-stepping schemes
- Dirichlet boundary conditions
- Vectorised NumPy operations for efficient Laplacian computation
- Transient vs. steady-state heat conduction