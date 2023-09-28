import numpy as np

import warnings
import time

#TODO
#IMPROVE RUNTIMES

#acceptance prob sometimes underflows, has no effect on calculations since underflow = 0
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")

def hamiltonian(r, omega):
    """
    Calculate the Hamiltonian of a quantum system.

    Args:
        r (np.ndarray): The positions of particles in the system.
        omega (float): The harmonic oscillator frequency.

    Returns:
        float: The Hamiltonian Operator
    """
    N = len(r) // dof
    
    kinetic_energy = -0.5 * np.sum(np.gradient(np.gradient(r)))
    potential_energy =  0.5 * omega**2 * np.sum(r**2)
    
    interaction_energy = 0
    for i in range(N):
        for j in range(i + 1, N):
            interaction_energy += 1.0 / np.linalg.norm(r[i*dof:(i+1)*dof] - r[j*dof:(j+1)*dof])
    
    return kinetic_energy + potential_energy + interaction_energy

class RBM:
    """
    Restricted Boltzmann Machine (RBM) class.

    Args:
        num_visible (int): Number of visible units (input layer).
        num_hidden (int): Number of hidden units.
    """
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = np.random.randn(num_visible, num_hidden)
        self.a = np.zeros(num_visible)
        self.b = np.zeros(num_hidden)

    def sigmoid(self, x):
        """
        Compute the sigmoid function.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Sigmoid-transformed values.
        """
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, v):
        """
       Perform a forward pass through the RBM.

       Args:
           v (np.ndarray): Visible layer activations.

       Returns:
           np.ndarray: Hidden layer activations.
       """
        p_h_given_v = self.sigmoid(np.dot(v, self.W) + self.b)
        h = np.random.binomial(1, p_h_given_v)
        return h

    def backward(self, h):
        """
        Perform a backward pass through the RBM.

        Args:
            h (np.ndarray): Hidden layer activations.

        Returns:
            np.ndarray: Visible layer activations.
        """
        p_v_given_h = self.sigmoid(np.dot(h, self.W.T) + self.a)
        v = np.random.binomial(1, p_v_given_h)
        return v

def metropolis_hastings_update(r, omega, num_samples):
    """
    Perform Metropolis-Hastings updates to generate samples.

    Args:
        r (np.ndarray): Initial positions of particles.
        omega (float): The harmonic oscillator frequency.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated samples.
    """
    samples = []
    current_r = np.random.randn(num_visible)
    current_energy = hamiltonian(current_r, omega)

    for _ in range(num_samples):
        proposed_r = current_r + 0.1 * np.random.randn(num_visible)
        proposed_energy = hamiltonian(proposed_r, omega)
        acceptance_prob = np.exp(proposed_energy - current_energy)

        if np.random.rand() < acceptance_prob:
            current_r = proposed_r
            current_energy = proposed_energy

        samples.append(current_r.copy())

    return np.array(samples)

def variational_monte_carlo(rbm, num_samples, num_iterations):
    """
    Perform Variational Monte Carlo (VMC) optimization to find the ground state energy.

    Args:
        rbm (RBM): Restricted Boltzmann Machine model.
        num_samples (int): Number of Metropolis-Hastings samples per iteration.
        num_iterations (int): Number of iterations.
    """
    learning_rate = 0.01
    rbm_params = [rbm.W, rbm.a, rbm.b]

    for _ in range(num_iterations):
        samples = metropolis_hastings_update(rbm_params, omega, num_samples)

        gradient = np.mean([hamiltonian(sample, omega) for sample in samples])

        for param in rbm_params:
            param += learning_rate * gradient * param
        

if __name__ == "__main__":
    t0 = time.time()
    runs = 1
    omega = 1.0
    num_particles = 2  
    dof = 2
    num_visible = num_particles * dof
    num_hidden=6
    num_samples = 2000
    num_iterations = 1000
    energy_storage = []
    verbose = False
    
    for k in range(runs):
        r = np.random.randn(num_visible)
        rbm = RBM(num_visible, num_hidden)
        
        variational_monte_carlo(rbm, num_samples, num_iterations) 
    
        ground_state_energy = []
        for _ in range(num_samples):
            sample = np.random.randn(num_visible)
            local_energy = hamiltonian(sample, omega)
            ground_state_energy.append(local_energy)
    
        estimated_energy = np.mean(ground_state_energy)
        if verbose:
            print(f"Energy: {estimated_energy:.3f} a.u.")
        energy_storage.append(estimated_energy)
    print(f"Mean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")

