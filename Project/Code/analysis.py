import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm

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
