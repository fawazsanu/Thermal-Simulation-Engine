# 2D Thermal Heat Dissipation Engine

## Project Overview
This project is a computational physics engine designed to simulate transient heat conduction across a 2D surface. By solving the Parabolic Partial Differential Equation (the Heat Equation) using numerical methods, this software models how thermal energy diffuses through a material over time.

## Mathematical Approach
The engine utilizes an explicit finite difference method (Forward Time, Centered Space). To ensure mathematical stability and prevent the simulation from diverging to infinity, the time step (`dt`) is dynamically calculated strictly based on the Von Neumann stability analysis constraint: 
`dt ≤ (dx²) / (4 * alpha)`

## Key Features
- **Vectorized Computations:** Replaced nested loops with optimized NumPy array slicing, pushing heavy matrix mathematics down to the C-level for rapid execution.
- **Dirichlet Boundary Conditions:** Supports fixed-temperature boundaries to simulate real-world environmental heat sinks and sources.
- **Thermal Visualization:** Generates accurate, color-mapped contour plots of the thermal gradients at various iteration milestones using Matplotlib.

## How to Run
1. Install dependencies:
   ```
   pip install -r requirements.txt
2. Run the simulation:
   ```
   python heat_solver_2d.py
