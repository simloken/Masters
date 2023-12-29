import numpy as np

import warnings

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
        self.W = 0.1 * np.random.randn(num_visible, num_hidden)
        self.a = np.zeros(num_visible)
        self.b = np.zeros(num_hidden)
        self.energy = None


    def rbm_energy(self, v):
        """
        Calculate the energy of the RBM for a given visible unit configuration.

        Args:
            v (np.ndarray): Visible unit configuration.

        Returns:
            float: Energy of the RBM for the given configuration.
        """
        energy = -np.sum(self.a * v) - np.sum(self.b * self.sigmoid(np.dot(v, self.W) + self.b))
        return energy
    
    
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
        p_v_given_h = np.dot(h, self.W.T) + self.a
        v = np.random.normal(p_v_given_h)
        return v
    
    def persistent_contrastive_divergence(self, v0, learning_rate, k=500):
        """
        Perform one step of Persistent Contrastive Divergence (PCD) learning.

        Args:
            v0 (np.ndarray): Input data for contrastive divergence.
            learning_rate (float): Learning rate for weight updates.
            k (int): Number of Gibbs sampling steps for CD.

        Returns:
            None
        """
        ph0 = self.sigmoid(np.dot(v0, self.W) + self.b)
        h0 = np.random.binomial(1, ph0) # Unused

        vk = v0.copy()
        for _ in range(k):
            phk = self.sigmoid(np.dot(vk, self.W) + self.b)
            hk = np.random.binomial(1, phk)
            pvk = np.dot(hk, self.W.T) + self.a
            vk = np.random.normal(pvk)

        self.W += learning_rate * (np.outer(v0, ph0) - np.outer(vk, phk))
        self.a += learning_rate * (v0 - vk)
        self.b += learning_rate * (ph0 - phk)



def metropolis_hastings_update(model, rbm, num_samples, num_visible, dof):
    """
    Perform Metropolis-Hastings updates to generate samples.

    Args:
        model (object): Object for passing the Hamiltonian.
        rbm (object): Object representing the Restricted Boltzmann Machine.
        num_samples (int): Number of samples to generate.
        num_visible (int): The number of visible units.
        dof (int): The degrees of freedom of the system.

    Returns:
        np.ndarray: Generated samples.
    """
    samples = []
    current_r = np.random.randn(num_visible)
    current_energy = model.hamiltonian(current_r, dof) + rbm.rbm_energy(current_r)

    for _ in range(num_samples):
        proposed_r = current_r + 0.005 * np.random.randn(num_visible)
        proposed_energy = model.hamiltonian(proposed_r, dof) + rbm.rbm_energy(proposed_r)
        acceptance_prob = np.exp(proposed_energy - current_energy)

        if np.random.rand() < acceptance_prob:
            current_r = proposed_r
            current_energy = proposed_energy

        samples.append(current_r.copy())
        if model.x_0:
            model.x_0 -= 0.0000004

    return np.array(samples)


def metropolis_hastings_spin_update(model, rbm, num_samples, num_visible, dof):
    """
    Perform Metropolis-Hastings updates to generate samples with 1/2 spins.

    Args:
        model (object): Object for passing the Hamiltonian.
        rbm (object): Object representing the Restricted Boltzmann Machine.
        num_samples (int): Number of samples to generate.
        num_visible (int): The number of visible units.
        dof (int): The degrees of freedom of the system.

    Returns:
        np.ndarray: Generated samples with 1/2 spins.
    """
    samples = []
    current_spin = np.random.choice([-0.5, 0.5], size=num_visible)
    current_energy = model.hamiltonian(current_spin, dof) + rbm.rbm_energy(current_spin)

    for _ in range(num_samples):
        proposed_spin = current_spin + 0.5 * np.random.choice([-1, 1], size=num_visible)
        proposed_energy = model.hamiltonian(proposed_spin, dof) + rbm.rbm_energy(proposed_spin)
        acceptance_prob = np.exp(proposed_energy - current_energy)

        if np.random.rand() < acceptance_prob:
            current_spin = proposed_spin
            current_energy = proposed_energy

        samples.append(current_spin.copy())

    return np.array(samples)

def variational_monte_carlo(model, rbm, num_visible, num_samples, num_iterations, dof):
    """
    Perform Variational Monte Carlo (VMC) optimization to find the ground state energy.

    Args:
        model (object): The hamiltonian model
        rbm (RBM): Restricted Boltzmann Machine model.
        num_visible (int): The number of visible units in the RBM model, num_particles x dof
        num_samples (int): Number of Metropolis-Hastings samples per iteration.
        num_iterations (int): Number of iterations.
        dof (int): The degrees of freedom of the system
    """
    learning_rate = 0.005
    rbm_params = [rbm.W, rbm.a, rbm.b]

    persistent_chains = [np.random.binomial(1, 0.5, num_visible) for _ in range(num_samples)]

    for i in range(num_iterations):
        if model.spin:
            samples = metropolis_hastings_spin_update(model, rbm, num_samples, num_visible, dof)
        else:
            samples = metropolis_hastings_update(model, rbm, num_samples, num_visible, dof)


        gradient = np.mean([np.gradient(sample) for sample in samples], axis=0)
        

        for param, grad, persistent_chain in zip(rbm_params, gradient, persistent_chains):
            rbm.persistent_contrastive_divergence(persistent_chain, learning_rate)
            persistent_chain[:] = rbm.backward(rbm.forward(persistent_chain))
            
        
    if not rbm.energy: #true energy
        rbm.energy = model.energy
    if model.x_0:
        print(model.x_0)
        
    return samples
    
