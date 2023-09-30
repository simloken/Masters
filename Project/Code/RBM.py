import numpy as np

import time
import warnings


#TODO
#IMPROVE RUNTIMES

#acceptance prob sometimes underflows, has no effect on calculations since underflow = 0
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow encountered")

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

def metropolis_hastings_update(hamiltonian, r, omega, num_samples, num_visible, dof):
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
    current_energy = hamiltonian(current_r, omega, dof)

    for _ in range(num_samples):
        proposed_r = current_r + 0.1 * np.random.randn(num_visible)
        proposed_energy = hamiltonian(proposed_r, omega, dof)
        acceptance_prob = np.exp(proposed_energy - current_energy)

        if np.random.rand() < acceptance_prob:
            current_r = proposed_r
            current_energy = proposed_energy

        samples.append(current_r.copy())

    return np.array(samples)

def variational_monte_carlo(hamiltonian, rbm, num_visible, num_samples, num_iterations, omega, dof):
    """
    Perform Variational Monte Carlo (VMC) optimization to find the ground state energy.

    Args:
        hamiltonian (function): The hamiltonian of the model
        rbm (RBM): Restricted Boltzmann Machine model.
        num_visible (int): The number of visible units in the RBM model, num_particles x dof
        num_samples (int): Number of Metropolis-Hastings samples per iteration.
        num_iterations (int): Number of iterations.
        omega (float): The harmonic oscillator frequency
        dof (int): The degrees of freedom of the system
    """
    learning_rate = 0.05
    rbm_params = [rbm.W, rbm.a, rbm.b]

    for _ in range(num_iterations):
        samples = metropolis_hastings_update(hamiltonian, rbm_params, num_visible, omega, num_samples, dof)

        gradient = np.mean([hamiltonian(sample, omega, dof) for sample in samples])

        for param in rbm_params:
            param += learning_rate * gradient * param
        
