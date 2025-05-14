import numpy as np
import matplotlib.pyplot as plt


def generate_moving_vortex(n_timesteps, nx, ny, lx=1, ly=1, period=100, plot_series=False, plot_interval=1):
    """
    Generate a moving vortex flow field based on the Lamb–Oseen vortex solution.
    The instantaneous velocity field is computed as:
    
        u(x,y,t) = - (Gamma/(2π)) * ( (y - y0(t)) / r² ) * [1 - exp(-r²/r_c²)]
        v(x,y,t) =   (Gamma/(2π)) * ( (x - x0(t)) / r² ) * [1 - exp(-r²/r_c²)]
        
    where r² = (x - x0(t))² + (y - y0(t))².
    """
    # Vortex parameters
    Gamma = 1.0
    r_c = 0.1
    x_center = 0.5
    y_center = 0.5
    r_move = 0.3  # amplitude of the center's circular motion

    data = np.zeros((n_timesteps, nx, ny))
    u_field = np.zeros((n_timesteps, nx, ny))
    v_field = np.zeros((n_timesteps, nx, ny))
    
    # Define physical domain: x in [0,lx], y in [0,ly]
    x = np.linspace(0, lx, nx)
    y = np.linspace(0, ly, ny)
    # Use 'ij' indexing: first index corresponds to x, second to y.
    Xgrid, Ygrid = np.meshgrid(x, y, indexing='ij')
    
    for t in range(n_timesteps):
        theta = 2 * np.pi * t / period
        x0 = x_center + r_move * np.cos(theta)
        y0 = y_center + r_move * np.sin(theta)
        
        dx = Xgrid - x0
        dy = Ygrid - y0
        r2 = dx**2 + dy**2
        # Avoid division by zero:
        r2[r2 == 0] = 1e-10
        
        factor = 1 - np.exp(-r2 / (r_c**2))
        u = - (Gamma / (2 * np.pi)) * (dy / r2) * factor
        v =   (Gamma / (2 * np.pi)) * (dx / r2) * factor
        
        u_field[t] = u
        v_field[t] = v
        data[t] = np.sqrt(u**2 + v**2)
    
    if plot_series:
        fig, axes = plt.subplots(1, n_timesteps, figsize=(4 * n_timesteps, 4))
        for t in range(0, n_timesteps, plot_interval):
            ax = axes[t]
            
            # Since data[t] is stored as (nx, ny) with x in axis 0 and y in axis 1,
            # we transpose it so that imshow interprets the first dimension as y (vertical)
            im = ax.imshow(data[t].T, origin='lower', extent=[0,lx,0,ly], cmap='viridis')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # The quiver uses Xgrid and Ygrid from meshgrid with indexing='ij',
            ax.quiver(Xgrid, Ygrid, u_field[t], v_field[t], color='white', scale=10, width=0.0007, pivot='mid')
            ax.set_title(f"Moving Vortex - Time Step {t}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        
        plt.tight_layout()
        plt.show()    
        
    return u_field, v_field


if __name__ == "__main__":
    _ = generate_moving_vortex(5, 100, 50, plot_series=True, plot_interval=1)