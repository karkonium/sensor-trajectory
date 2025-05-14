import numpy as np
import matplotlib.pyplot as plt


def generate_double_gyre_flow(n_timesteps, nx, ny, lx=2, ly=1,
                              A=0.1, epsilon=0.25, period=20, 
                              plot_series=False):
    """   
    The double gyre is defined on the domain x ∈ [0,2] and y ∈ [0,1]. Its velocity field is given by:
    
        f(x,t) = ε sin(ω t) x² + (1 - 2ε sin(ω t)) x
        u(x,y,t) = -π A sin(π f(x,t)) cos(π y)
        v(x,y,t) =  π A cos(π f(x,t)) sin(π y) [2ε sin(ω t) x + (1 - 2ε sin(ω t))]
    
    Parameters:
      n_timesteps : Number of timesteps.
      nx, ny : Number of spatial grid points in x and y directions.
      A : Amplitude of the velocity.
      epsilon : Strength of the time-periodic oscillation.
      period : Number of intervals for flow to repeat
      plot_series 
    
    Returns:
       u_field, v_field 
    """
    omega = 2*np.pi / period

    data    = np.zeros((n_timesteps, nx, ny))
    u_field = np.zeros((n_timesteps, nx, ny))
    v_field = np.zeros((n_timesteps, nx, ny))

    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)

    Xgrid, Ygrid = np.meshgrid(x, y, indexing='ij')
    
    for t in range(n_timesteps):
        # Time variable (assuming unit time steps)
        time = t
        sin_omega_t = np.sin(omega * time)
        
        # Define the function f(x,t) and its derivative with respect to x:
        # f(x,t) = ε sin(ω t) x² + (1 - 2ε sin(ω t)) x
        f = epsilon * sin_omega_t * Xgrid**2 + (1 - 2*epsilon*sin_omega_t) * Xgrid
        dfdx = 2 * epsilon * sin_omega_t * Xgrid + (1 - 2*epsilon*sin_omega_t)
        
        # Compute velocity components
        u = - np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * Ygrid)
        v =   np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * Ygrid) * dfdx
        
        u_field[t] = u
        v_field[t] = v
        data[t] = np.sqrt(u**2 + v**2)
    
    if plot_series:
        fig, axes = plt.subplots(1, n_timesteps, figsize=(4 * n_timesteps, 4))
        for t in range(0, n_timesteps, 1):
            ax = axes[t]
            
            # Plot magnitute
            # im = ax.imshow(data[t].T, origin='lower', extent=[0,lx,0,ly], cmap='viridis')
            # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Plot direction
            ax.quiver(Xgrid, Ygrid, u_field[t], v_field[t], color='black', 
                       scale_units='xy', scale=10, width=0.005, pivot='mid')
            
            ax.set_title(f"Double Gyre Flow - Time Step {t}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        
        plt.tight_layout()
        plt.show()
    
    return u_field, v_field


if __name__ == "__main__":
    # Generate the double gyre flow data over 50 timesteps on a 100x50 grid.
    _ = generate_double_gyre_flow(
        n_timesteps=6, nx=30, ny=15, lx=2, ly=1,
        A=0.1, epsilon=0.5, 
        period=6,
        plot_series=True
    )
