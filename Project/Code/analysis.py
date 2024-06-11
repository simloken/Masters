import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import gaussian_kde, norm
import matplotlib.patheffects as path_effects

import os

def relative_error(measured, exact):
    return (measured-exact)/exact

def plot_particle_density(positions, dof):
    if dof not in [1, 2, 3]:
        raise ValueError("dof must be 1, 2, or 3")
        
    if dof == 1:
        num_samples, num_particles = positions.shape

        plt.figure(figsize=(10, 5))
        color = 'k'

        for i in range(num_particles):
            mean = np.mean(positions[:, i])
            std_dev = np.std(positions[:, i])

            x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
            pdf = norm.pdf(x, loc=mean, scale=std_dev)

            plt.fill_between(x, 0, pdf, alpha=0.5, label=f"Particle {i + 1}", color=color)

        plt.xlabel("Position")
        plt.ylabel("Probability")
        plt.title("Particle Position Probability (1D)")
        #plt.legend()
        
    else:
        if dof == 2:
            x, y = positions[:, 0], positions[:, 1]
            kde = gaussian_kde([x, y])
            x_grid, y_grid = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
            z = kde(np.vstack([x_grid.ravel(), y_grid.ravel()]))
            z = z.reshape(x_grid.shape)

            plt.imshow(z, cmap="viridis", extent=(min(x), max(x), min(y), max(y)), origin="lower")
            plt.colorbar(label="Density")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.title("Particle Position Density (2D)")
        else:
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
            kde = gaussian_kde([x, y, z])
            density = kde(positions.T)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z, c=density, cmap="viridis")
            ax.set_xlabel("X Position")
            ax.set_ylabel("Y Position")
            ax.set_zlabel("Z Position")
            ax.set_title("Particle Position Density (3D)")

    plt.show()
    
    
def plot_wavefunction(psi, name, dof, particles):
    import torch
    if dof == 1:
        
        total_splits = dof * particles
        x_values_list = []
        split_range = np.linspace(-5, 5, total_splits+1)

        for i in range(total_splits):
            x_values_i = torch.linspace(split_range[i], split_range[i+1], 400).view(-1, 1)
            x_values_list.append(x_values_i)

        x_values = torch.cat(x_values_list, dim=1)
        psi_values = torch.square(psi(x_values))
        print(x_values)
        print(psi_values)
        plt.figure(figsize=(10, 6))
        plt.plot(x_values.numpy(), psi_values.numpy(), label='Wavefunction')
        plt.xlabel('x')
        plt.ylabel(r'$\Psi(x)$')
        plt.title(rf'Wavefunction $\Psi(x)$ for {name}')
        plt.legend()
        plt.show()



def sample_distribution_history(data, name, dof, bins=50, pause_duration=5):
    """
    Creates an animated plot of sample distributions over iterations and saves it as a GIF.
    
    Parameters:
    data (numpy array): A 3D numpy array of shape (iterations, samples, dof * num_particles).
    name (str): The name to use for saving the GIF file.
    dof (int): Degrees of freedom (1 for 1D plotting, 2 for 2D plotting, 3D pending implementation).
    bins (int): Number of bins for the histogram or heatmap.
    pause_duration (int): Duration to pause at the first and last frames (in frames).
    """
    
    print('Generating sample distribution history plot...')
    iterations, samples, total_dof = data.shape
    num_particles = total_dof // dof
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if dof == 1:
        def reshape_data_1d(data):
            reshaped_data = data.reshape(iterations, num_particles, samples)
            return reshaped_data
        
        reshaped_data = reshape_data_1d(data)
        
        # Generate a grid for KDE evaluation
        x_min, x_max = np.min(reshaped_data), np.max(reshaped_data)
        x_grid = np.linspace(x_min, x_max, bins)

        def compute_kde_1d(data):
            kde = gaussian_kde(data)
            kde_values = kde(x_grid)
            return kde_values

        # Initial and final KDE values for the initial and final frames
        initial_data = reshaped_data[0].reshape(-1)
        final_data = reshaped_data[-1].reshape(-1)
        initial_kde_values = [compute_kde_1d(initial_data[i*samples:(i+1)*samples]) for i in range(num_particles)]
        final_kde_values = [compute_kde_1d(final_data[i*samples:(i+1)*samples]) for i in range(num_particles)]

        # Calculate the maximum KDE values for initial and final distributions
        max_initial_kde = np.max(initial_kde_values, axis=0)
        max_final_kde = np.max(final_kde_values, axis=0)
        
        # Plot initial and final KDE
        ax.plot(x_grid, max_initial_kde, color='red', label='Initial Distribution')
        ax.plot(x_grid, max_final_kde, color='green', label='Final Distribution')

        # Initial frame KDE for current distributions
        current_kde_values = [compute_kde_1d(reshaped_data[0, i, :]) for i in range(num_particles)]
        max_current_kde = np.max(current_kde_values, axis=0)
        line, = ax.plot(x_grid, max_current_kde, color='blue')
        global fill
        fill = ax.fill_between(x_grid, max_current_kde, color='blue', alpha=0.5)

        def init():
            global fill
            line.set_ydata(max_current_kde)
            fill.remove()
            fill = ax.fill_between(x_grid, max_current_kde, color='blue', alpha=0.5)
            iteration_text.set_text('')
            return line, iteration_text, fill

        def update(frame):
            global fill
            current_data = reshaped_data[frame % iterations]
            kde_values = [compute_kde_1d(current_data[i, :]) for i in range(num_particles)]
            max_kde_values = np.max(kde_values, axis=0)
            line.set_ydata(max_kde_values)
            fill.remove()
            fill = ax.fill_between(x_grid, max_kde_values, color='blue', alpha=0.5)
            iteration_text.set_text(f'Iteration: {frame % iterations}')
            return line, iteration_text, fill
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Probability Density')
        
        # Add iteration counter text with white color and black edge
        iteration_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                                 verticalalignment='top', color='white', path_effects=[
                                     path_effects.withStroke(linewidth=3, foreground='black')])
        
        # Add legend
        ax.legend()
        
        # Create frame sequence with pauses
        frames = [0] * pause_duration + list(range(1, iterations - 1)) + [iterations - 1] * pause_duration
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                      blit=True, repeat=False)
    
    elif dof == 2:
        def reshape_data_2d(data):
            reshaped_data = data.reshape(iterations, num_particles, samples, dof)
            return reshaped_data
        
        reshaped_data = reshape_data_2d(data)

        x_min, x_max = np.min(reshaped_data[:,:,:,0]), np.max(reshaped_data[:,:,:,0])
        y_min, y_max = np.min(reshaped_data[:,:,:,1]), np.max(reshaped_data[:,:,:,1])
        x_grid = np.linspace(x_min, x_max, bins)
        y_grid = np.linspace(y_min, y_max, bins)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([x_mesh.ravel(), y_mesh.ravel()])

        def compute_kde_2d(data):
            kde = gaussian_kde(data.T)
            kde_values = kde(grid_points).reshape(bins, bins)
            return kde_values

        initial_data = reshaped_data[0].reshape(-1, dof)
        final_data = reshaped_data[-1].reshape(-1, dof)
        initial_kde_values = compute_kde_2d(initial_data)
        final_kde_values = compute_kde_2d(final_data)

        initial_contour = ax.contour(x_grid, y_grid, initial_kde_values, colors='red', alpha=0.5)
        final_contour = ax.contour(x_grid, y_grid, final_kde_values, colors='green', alpha=0.5)

        current_data = reshaped_data[0].reshape(-1, dof)
        current_kde_values = compute_kde_2d(current_data)
        heatmap = ax.imshow(current_kde_values.T, extent=[x_min, x_max, y_min, y_max],
                            origin='lower', aspect='auto', cmap='viridis', interpolation='nearest')
        
        cbar = plt.colorbar(heatmap, ax=ax)
        cbar.set_label('Probability Density')

        def init():
            heatmap.set_array(current_kde_values.T)
            iteration_text.set_text('')
            return heatmap, iteration_text
        
        def update(frame):
            current_data = reshaped_data[frame % iterations].reshape(-1, dof)
            kde_values = compute_kde_2d(current_data)
            heatmap.set_array(kde_values.T)
            iteration_text.set_text(f'Iteration: {frame % iterations}')
            return heatmap, iteration_text
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        
        iteration_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, fontsize=12, 
                                 verticalalignment='top', color='white', path_effects=[
                                     path_effects.withStroke(linewidth=3, foreground='black')])
        
        red_line = plt.Line2D([0], [0], color='red', lw=2, label='Initial Distribution')
        green_line = plt.Line2D([0], [0], color='green', lw=2, label='Final Distribution')
        ax.legend(handles=[red_line, green_line])
        
        frames = [0] * pause_duration + list(range(1, iterations - 1)) + [iterations - 1] * pause_duration
        ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init,
                                      blit=True, repeat=False)
    
    save_dir = os.path.join('..', 'Figures', 'Animations')
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'{name}_sample_history.gif')
    ani.save(save_path, writer='pillow', fps=10, dpi=80)
    plt.close(fig)