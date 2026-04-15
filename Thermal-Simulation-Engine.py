import numpy as np
import matplotlib.pyplot as plt

def simulate_heat_transfer():
    
    # PHYSICAL & NUMERICAL PARAMETERS
    grid_size = 50          # 50x50 grid resolution
    physical_length = 1.0   # Physical size of the plate (i.e., 1 meter)
    dx = physical_length / grid_size # Spatial step size (dx = dy)
    dy = dx
    
    alpha = 0.0001          # Thermal diffusivity of the material (i.e., Steel)
    
    # Calculate stable time step based on the Von Neumann stability analysis
    # For 2D explicit methods, dt must be <= dx^2 / (4 * alpha)
    dt = (dx**2) / (4 * alpha) * 0.9  # 0.9 is a safety factor to prevent exponential growth
    
    total_time_steps = 3000 # How long the simulation runs

   
    # 2. INITIALIZE THE GRID & BOUNDARY CONDITIONS
    # Create a 2D matrix filled with room temperature (i.e., 20°C)
    u = np.full((grid_size, grid_size), 20.0)
    
    # Apply Dirichlet Boundary Conditions (Constant heat sources)
    u[0, :] = 100.0   # Top edge is held at 100°C (Boiling water)
    u[-1, :] = 0.0    # Bottom edge is held at 0°C (Ice)
    u[:, 0] = 20.0    # Left edge insulated/room temp
    u[:, -1] = 20.0   # Right edge insulated/room temp

    # VISUALIZATION
    snapshots = []
    capture_intervals = [0, 100, 1000, 2999]

    print("Starting finite difference iterations...")

    # THE NUMERICAL SOLVER (TIME ITERATION)
    for step in range(total_time_steps):
        if step in capture_intervals:
            snapshots.append(u.copy())
            
        # Create a copy to store the next time step's values
        u_next = u.copy()
        
        # VECTORIZED LAPLACIAN COMPUTATION
        # The matrix is sliced to calculate the gradients instantly (d^2u/dx^2 + d^2u/dy^2)
        d2u_dx2 = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / (dx**2)
        d2u_dy2 = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / (dy**2)
        
        # Explicit Forward Euler Time Integration
        u_next[1:-1, 1:-1] = u[1:-1, 1:-1] + alpha * dt * (d2u_dx2 + d2u_dy2)
        
        # Update the grid
        u = u_next

    print("Simulation complete. Rendering thermal gradients...")

    # RENDER THE THERMAL CONTOUR MAPS
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # Plot the matrix as a colored heatmap
        im = ax.imshow(snapshots[i], cmap='inferno', vmin=0, vmax=100, origin='lower')
        ax.set_title(f'Thermal State at Iteration {capture_intervals[i]}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        fig.colorbar(im, ax=ax, label='Temperature (°C)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_heat_transfer()