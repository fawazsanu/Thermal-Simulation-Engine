# Thermal Simulation Engine

A 2D finite difference heat transfer simulator using the explicit Forward Euler method. Models how temperature distributes and evolves across a solid plate over time, with fixed boundary conditions and vectorised NumPy computation.

---

## Overview

The simulation solves the **2D heat equation** (parabolic PDE) numerically on a uniform grid:

```
∂u/∂t = α · (∂²u/∂x² + ∂²u/∂y²)
```

Where `u(x, y, t)` is temperature and `α` is the thermal diffusivity of the material. The solver uses a vectorised **explicit finite difference scheme** (FTCS — Forward Time, Centred Space), with a time step chosen to satisfy the Von Neumann stability criterion.

Results are rendered as a 2×2 grid of thermal contour maps showing the temperature field at four points in time.

---

## Physical Setup

| Parameter         | Value        | Notes                                    |
|-------------------|--------------|------------------------------------------|
| Grid resolution   | 50 × 50      | Uniform spatial discretisation           |
| Plate size        | 1.0 × 1.0 m  | Physical domain                          |
| Spatial step `dx` | 0.02 m       | `physical_length / grid_size`            |
| Thermal diffusivity `α` | 1×10⁻⁴ m²/s | Approximate value for steel        |
| Time step `dt`    | 0.9 s        | 90% of the Von Neumann stability limit   |
| Total steps       | 3,000        | ~2,700 simulated seconds (~45 min)       |

### Boundary Conditions (Dirichlet)

| Edge        | Temperature | Physical analogy       |
|-------------|-------------|------------------------|
| Top         | 100 °C      | Boiling water contact  |
| Bottom      | 0 °C        | Ice contact            |
| Left / Right | 20 °C      | Ambient / insulated    |

Interior cells initialise at 20 °C (room temperature).

---

## Numerical Method

**Stability criterion (Von Neumann analysis for 2D explicit schemes):**

```
dt ≤ dx² / (4 · α)
```

With `dx = 0.02` and `α = 1×10⁻⁴`, the maximum stable time step is **1.0 s**. The simulation uses `dt = 0.9 s` (safety factor 0.9), ensuring unconditional stability throughout the run.

**Vectorised Laplacian:** Rather than looping over every cell, the spatial second derivatives are computed via NumPy array slicing across the interior:

```python
d2u_dx2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
d2u_dy2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dy**2
u_next[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * dt * (d2u_dx2 + d2u_dy2)
```

This is equivalent to applying the standard 5-point stencil across all interior cells simultaneously.

---

## Requirements

```
numpy
matplotlib
```

Install with:

```bash
pip install numpy matplotlib
```

Python 3.7+ recommended.

---

## Usage

```bash
python Thermal-Simulation-Engine.py
```

The solver prints progress messages, then opens an interactive matplotlib window displaying four snapshots:

| Snapshot | Iteration | Simulated Time |
|----------|-----------|----------------|
| 1        | 0         | 0 s (initial)  |
| 2        | 100       | 90 s           |
| 3        | 1,000     | 900 s          |
| 4        | 2,999     | ~2,699 s       |

The `inferno` colormap is used, spanning 0 °C (black) to 100 °C (bright yellow).

---

## Expected Output

At **iteration 0**, the grid shows sharp discontinuities at the boundaries with the interior at uniform 20 °C. By **iteration 2,999**, the temperature field approaches a smooth steady-state gradient — warmer near the top edge, cooler near the bottom, with the left/right edges holding at 20 °C.

---

## Limitations & Potential Extensions

- **Explicit scheme only** — larger grids or higher diffusivity values require proportionally smaller time steps. An implicit (Crank-Nicolson) solver would remove this constraint
- **Homogeneous material** — `α` is uniform across the grid; real plates may have spatially varying properties
- **Fixed boundaries** — Neumann (flux) boundary conditions (e.g., insulated edges with zero gradient) are not implemented
- **No convergence check** — the simulation runs for a fixed number of steps rather than stopping when steady state is reached
- **2D only** — extending to 3D would require a `(N, N, N)` array and adjusted Laplacian stencil
