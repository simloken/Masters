import torch

import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax

from energies import Energies

class NN:
    def __init__(self, hamiltonian, params):
        self.hamiltonian = hamiltonian
        self.params = params
        self.first_pass = True
        self.x_0 = False
        
        accepted_hamiltonians = ['harmonic_oscillator',
                                 'two_fermions', 'calogero_sutherland',
                                 'ising', 'heisenberg']
        
        if hamiltonian not in accepted_hamiltonians:
            raise ValueError('Unrecognized Hamiltonian, try: ', accepted_hamiltonians)
        
        if hamiltonian == 'harmonic_oscillator':
            self.hamiltonian = self.harmonic_oscillator
            self.name = 'harmonic_oscillator'
            self.has_plots = False
            self.spin = False
            self.symmetric = True
            if self.params == 'default':
                self.params = 1
        
        elif hamiltonian == 'two_fermions':
            self.hamiltonian = self.two_fermions
            self.name = 'two_fermions'
            self.has_plots = False
            self.spin = False
            self.symmetric = False
            if self.params == 'default':
                self.params = 1
            
        elif hamiltonian == 'calogero_sutherland':
            self.hamiltonian = self.calogero_sutherland
            self.x_0 = 0.3
            self.x_0_initial = self.x_0
            self.x_0_minimum = 0.1
            self.name = 'calogero_sutherland'
            self.has_plots = True
            self.spin = False
            self.symmetric = True
            if self.params == 'default':
                self.params = [1,2]
            
        elif hamiltonian == 'ising':
            self.hamiltonian = self.ising
            self.name = 'ising'
            self.has_plots = False
            self.spin = True
            self.symmetric = '!' #neither symmetric nor antisymmetric
            if self.params == 'default':
                self.params = [-1,-1]
            
        elif hamiltonian == 'heisenberg':
            self.hamiltonian = self.heisenberg
            self.name = 'heisenberg'
            self.has_plots = False
            self.spin = True
            self.symmetric = '!' #neither symmetric nor antisymmetric
            if self.params == 'default':
                self.params = []
                
            
        self.kinetic_energy = []
        self.potential_energy = []
            

    def harmonic_oscillator(self, psi, positions):
        """
        Calculate the Hamiltonian operator for a 1D harmonic oscillator.

        Args:
            psi (callable): A PyTorch neural network estimating the wavefunction.
            positions (list of torch.Tensor): List of positional tensors for each particle.

        Returns:
            torch.Tensor: The Hamiltonian operator.
        """
        x = positions
        omega = self.params

        x.requires_grad_(True)
    
        psi_x = psi(x)
        
        psi_x_grad = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        
        psi_x_grad2 = torch.autograd.grad(psi_x_grad, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        
        kinetic_energy = - 1/2 * psi_x_grad2
        
        potential_energy = 0.5*omega**2 * x**2 * psi_x
        
        hamiltonian_operator = kinetic_energy + potential_energy

        if self.first_pass:
            self.energy = Energies.harmonic_oscillator(omega)
            self.first_pass = False
        
        return hamiltonian_operator

    def two_fermions(self, psi, positions):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            psi (callable): A PyTorch neural network estimating the wavefunction.
            positions (list of torch.Tensor): List of positional tensors for each particle.

        Returns:
            torch.Tensor: The Hamiltonian operator.
        """
        
        x = positions
        omega = self.params

        x.requires_grad_(True)
        
        psi_x = psi(x)
        
        psi_x_grad = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
        psi_x_laplacian = torch.autograd.grad(psi_x_grad, x, grad_outputs=torch.ones_like(psi_x_grad), create_graph=True)[0]
        
        kinetic_energy = -0.5 * psi_x_laplacian.sum(dim=1, keepdim=True)
        
        r_squared = (x**2).sum(dim=1, keepdim=True)
        potential_energy = 0.5 * omega**2 * r_squared * psi_x
        
        r1 = x[:, :2]
        r2 = x[:, 2:]  
        r12 = torch.norm(r1 - r2, dim=1, keepdim=True)
        interaction_energy = (1 / r12) * psi_x
        
        H_psi = kinetic_energy + potential_energy + interaction_energy
        
        if self.first_pass:
            self.energy = Energies.two_fermions()
            self.first_pass = False


        return H_psi

    def calogero_sutherland(self, psi, positions):
        """
        Calculate the Hamiltonian operator for the Calogero-Sutherland model.

        Args:
            psi (callable): A PyTorch neural network estimating the wavefunction.
            positions (list of torch.Tensor): List of positional tensors for each particle.

        Returns:
            torch.Tensor: The Hamiltonian operator.
        """
        samples, N = positions.shape
        omega, beta = self.params
        x_0 = self.x_0

        x = positions
        x.requires_grad_(True)
        psi_x = psi(x)
        gradient_psi_x = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
        gradient_psi_x_i = torch.chunk(gradient_psi_x, N, dim=1)

        kinetic_energy = sum(-0.5 * torch.sum(grad ** 2, dim=1) for grad in gradient_psi_x_i)
        potential_energy = 0.5 * omega ** 2 * torch.sum(x ** 2, dim=1) * psi_x.squeeze()
        H_Psi = kinetic_energy + potential_energy

        x_expanded_1 = x.unsqueeze(2)
        x_expanded_2 = x.unsqueeze(1)
        x_ij = x_expanded_1 - x_expanded_2
        term = beta * (beta - 1) * torch.tanh(x_ij / x_0) ** 2 / (x_ij ** 2 + 1e-8)
        mask = torch.triu(torch.ones((N, N)), diagonal=1)
        term = term * mask
        interaction_energy = torch.sum(term, dim=[1, 2])

        H_Psi += interaction_energy * psi_x.squeeze()

        if self.first_pass:
            self.energy = Energies.calogero_sutherland(N, omega, beta)
            self.first_pass = False


        return H_Psi

    def ising(self, psi, spins):
        """
        Calculate the Hamiltonian operator for the 1D Transverse Ising Model.

        Args:
            psi (callable): A PyTorch neural network estimating the wavefunction.
            spins (list of torch.Tensor): List of spin configurations for each particle.

        Returns:
            torch.Tensor: The Hamiltonian operator.
        """
        Gamma, V = self.params
        samples, N = spins.shape

        interaction_term = V * torch.sum(spins * torch.roll(spins, shifts=-1, dims=1), dim=1)

        transverse_field_term = 0
        for i in range(N):
            flipped_spins = spins.clone()
            flipped_spins[:, i] *= -1
            transverse_field_term += Gamma * (psi(flipped_spins) / psi(spins))

        hamiltonian_operator = interaction_term + transverse_field_term

        if self.first_pass:
            self.energy = Energies.ising(N)
            self.first_pass = False

        hamiltonian_operator = hamiltonian_operator.unsqueeze(0)

        return hamiltonian_operator

    def heisenberg(self, psi, spins):
        """
        Calculate the Hamiltonian operator for the 1/2 spin Heisenberg antiferromagnetic chain.

        Args:
            psi (callable): A PyTorch neural network estimating the wavefunction.
            spins (list of torch.Tensor): List of spin configurations for each particle.

        Returns:
            torch.Tensor: The Hamiltonian operator.
        """
        samples, N = spins.shape

        interaction_term = 0
        for i in range(N):
            flipped_spins_x = spins.clone()
            flipped_spins_x[:, i] *= -1
            flipped_spins_x[:, (i + 1) % N] *= -1

            flipped_spins_y = spins.clone().to(dtype=torch.complex64)
            flipped_spins_y[:, i] *= -1j
            flipped_spins_y[:, (i + 1) % N] *= 1j

            interaction_term += (
                (psi(flipped_spins_x) / psi(spins))
                + (psi(flipped_spins_y) / psi(spins))
                + (spins[:, i] * spins[:, (i + 1) % N])
            )

        hamiltonian_operator = interaction_term

        if self.first_pass:
            self.energy = Energies.heisenberg(N)
            self.first_pass = False

        hamiltonian_operator = hamiltonian_operator.unsqueeze(0)
        
        return hamiltonian_operator
   
    
    
    
    def plot_energy_components(self):
        plt.plot(self.kinetic_energy, label='T')
        plt.plot(self.potential_energy, label='V')
        plt.title('DEBUG PLOT - ENERGY COMPONENT TRACKING')
        plt.legend()
        plt.show()
        
    
    def lipkin(self, psi, spins):
        """
        Calculate the Hamiltonian operator for the Lipkin model.
    
        Args:
            psi (callable): A TensorFlow neural network estimating the wavefunction.
            spins (list of tf.Tensor): List of spin tensors for each particle.
            
        Returns:
            tf.Tensor: The Hamiltonian operator.
        """
        
        d = self.params['d']  # single particle energies
        g = self.params['g']  # pairing strengths
    
        hamiltonian_operator = 0
    
        for p in range(len(spins)):
            spin_p = spins[p]
    
            N_p = tf.reduce_sum(spin_p)
    
            hamiltonian_operator += d[p] * N_p
    
            for q in range(len(spins)):
                spin_q = spins[q]
    
                A_dagger_p = create(spin_p)
                A_q = annihilate(spin_q)
    
                hamiltonian_operator -= g[p][q] * A_dagger_p * A_q
    
        return hamiltonian_operator

def create(spin):
    """
    Fermionic creation operator.

    Args:
        spin (tf.Tensor): Spin tensor for a particle.
        
    Returns:
        tf.Tensor: New spin tensor with energy level raised.
    """
    # TODO: Implement this function... maybe

def annihilate(spin):
    """
    Fermionic annihilation operator.

    Args:
        spin (tf.Tensor): Spin tensor for a particle.
        
    Returns:
        tf.Tensor: New spin tensor with energy level lowered.
    """
    # TODO: Implement this function... maybe
    
    
class RBM:
    def __init__(self, hamiltonian, params):
        self.hamiltonian = hamiltonian
        self.params = params
        self.first_pass = True
        self.x_0 = False
        
        accepted_hamiltonians = ['harmonic_oscillator',
                                 'two_fermions', 'calogero_sutherland',
                                 'ising', 'heisenberg']
        
        if hamiltonian not in accepted_hamiltonians:
            raise ValueError('Unrecognized Hamiltonian, try: ', accepted_hamiltonians)
        
        if hamiltonian == 'harmonic_oscillator':
            self.hamiltonian = self.harmonic_oscillator
            self.name = 'harmonic_oscillator'
            self.has_plots = False
            self.spin = False
            if self.params == 'default':
                self.params = 1
        
        elif hamiltonian == 'two_fermions':
            self.hamiltonian = self.two_fermions
            self.name = 'two_fermions'
            self.has_plots = False
            self.spin = False
            if self.params == 'default':
                self.params = 1
            
        elif hamiltonian == 'calogero_sutherland':
            self.hamiltonian = self.calogero_sutherland
            self.x_0 = 0.3
            self.x_0_initial = self.x_0
            self.x_0_minimum = 0.1
            self.name = 'calogero_sutherland'
            self.has_plots = True
            self.spin = False
            if self.params == 'default':
                self.params = [1,2]
            
        elif hamiltonian == 'ising':
            self.hamiltonian = self.ising
            self.name = 'ising'
            self.has_plots = False
            self.spin = True
            if self.params == 'default':
                self.params = [-1,-1]
            
        elif hamiltonian == 'heisenberg':
            self.hamiltonian = self.heisenberg
            self.name = 'heisenberg'
            self.has_plots = False
            self.spin = True
            if self.params == 'default':
                self.params = []
                
            
    def harmonic_oscillator(self, psi, positions):
        """
        Calculate the Hamiltonian operator for a 1D harmonic oscillator.

        Args:
            psi (callable): A function estimating the wavefunction.
            positions (jnp.ndarray): Positional tensors for each particle.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        x = positions
        omega = self.params
        psi_x = psi(x, 1)
        gradient_psi_x = jax.vmap(jax.grad(lambda x_i: psi(x_i, 1, grad=True).squeeze()))(x)
    
        kinetic_energy = -0.5 * jnp.square(gradient_psi_x)

        potential_energy = 0.5 * omega**2 * jnp.square(x)*jnp.squeeze(psi_x)
        
        
        hamiltonian_operator = kinetic_energy + potential_energy
        

        if self.first_pass:
            self.energy = Energies.harmonic_oscillator(omega)
            self.first_pass = False
            

        hamiltonian_operator = jnp.expand_dims(hamiltonian_operator, axis=0)
                
        return hamiltonian_operator

    
    
    def two_fermions(self, psi, positions):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            psi (callable): A function estimating the wavefunction.
            positions (jnp.ndarray): Positional tensors for each particle.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        x = positions
        N, dof_times_particles = x.shape
        dof = dof_times_particles//2
        x1, x2 = jnp.split(x, 2, axis=1)
        omega = self.params

        psi_x = psi(x, 2)
            
        gradient_psi_x = jax.vmap(jax.grad(lambda x_i: psi(x_i, 1, grad=True).squeeze()))(x)
        
        gradient_psi_x_i = jnp.split(gradient_psi_x, 2, axis=1)

        kinetic_energy_1 = -0.5 * jnp.sum(jnp.square(gradient_psi_x_i[0]), axis=1)
        kinetic_energy_2 = -0.5 * jnp.sum(jnp.square(gradient_psi_x_i[1]), axis=1)


        potential_energy_1 = 0.5 * omega**2 * jnp.sum(jnp.square(x1), axis=1)*jnp.sum(psi_x)
        potential_energy_2 = 0.5 * omega**2 * jnp.sum(jnp.square(x2), axis=1)*jnp.sum(psi_x)

        epsilon = 1e-8
        r_ij = jnp.linalg.norm(x1[:, jnp.newaxis, :] - x2, axis=2)
        interaction_energy = psi_x * jnp.sum(1.0 / (r_ij + epsilon), axis=1) / N

        hamiltonian_operator = (
            kinetic_energy_1 + kinetic_energy_2 + potential_energy_1 + potential_energy_2 + interaction_energy
        )

        if self.first_pass:
            self.energy = Energies.two_fermions()
            self.first_pass = False

        hamiltonian_operator = jnp.expand_dims(hamiltonian_operator, axis=0)
        
        
        return hamiltonian_operator
    
    def calogero_sutherland(self, psi, positions):
        """
        Calculate the Hamiltonian operator for the Calogero-Sutherland model.

        Args:
            psi (callable): A function estimating the wavefunction.
            positions (jnp.ndarray): Positional tensors for each particle.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        samples, N = positions.shape
        omega, beta = self.params
        x_0 = self.x_0
        x = positions

        psi_x = psi(x, 1)
        gradient_psi_x = jax.vmap(jax.grad(lambda x_i: psi(x_i, 1, grad=True).squeeze()))(x)
        gradient_psi_x_i = jnp.split(gradient_psi_x, N, axis=1)

        kinetic_energy = 0
        for i in range(N):
            kinetic_energy += -0.5 * jnp.sum(jnp.square(gradient_psi_x_i[i]), axis=1)

        potential_energy = 0.5 * omega**2 * jnp.sum(jnp.square(x), axis=1) * psi_x
        hamiltonian_operator = kinetic_energy + potential_energy

        x_expanded_1 = jnp.expand_dims(x, axis=2)
        x_expanded_2 = jnp.expand_dims(x, axis=1)
        x_ij = x_expanded_1 - x_expanded_2

        term = beta * (beta - 1) * jnp.square(jnp.tanh(x_ij / x_0)) / (jnp.square(x_ij) + 1e-8)
        mask = jnp.tri(N, N, -1, dtype=jnp.float32)  # upper triangular part is 0
        term = term * mask

        interaction_energy = jnp.sum(term, axis=[1, 2])
        hamiltonian_operator += interaction_energy * psi_x

        if self.first_pass:
            self.energy = Energies.calogero_sutherland(N, omega, beta)
            self.first_pass = False

        hamiltonian_operator = jnp.expand_dims(hamiltonian_operator, axis=0)
        return hamiltonian_operator
    
    def ising(self, psi, spins):
        Gamma, V = self.params

        samples, N = spins.shape

        interaction_term = V * jnp.sum(spins * jnp.roll(spins, -1, axis=1), axis=1)

        def transverse_field_term_component(i, spins):
            flipped_spins = spins.at[:, i].set(-spins[:, i])
            return Gamma * (psi(flipped_spins, 1) / psi(spins, 1))

        transverse_field_term = jnp.sum(jax.vmap(transverse_field_term_component, in_axes=(0, None))(jnp.arange(N), spins), axis=0)

        hamiltonian_operator = interaction_term + transverse_field_term

        if self.first_pass:
            self.energy = Energies.ising(N)
            self.first_pass = False

        hamiltonian_operator = jnp.expand_dims(hamiltonian_operator, axis=0)

        return hamiltonian_operator
    
    def heisenberg(self, psi, spins):
        samples, N = spins.shape

        def interaction_term_component(i, spins):
            
            flipped_spins_x = spins.at[:, i].set(-spins[:, i])
            flipped_spins_x = flipped_spins_x.at[:, (i + 1) % N].set(-spins[:, (i + 1) % N])
            
            flipped_spins_y = spins.astype(jnp.complex64)
            flipped_spins_y = flipped_spins_y.at[:, i].set(-1j * spins[:, i])
            flipped_spins_y = flipped_spins_y.at[:, (i + 1) % N].set(1j * spins[:, (i + 1) % N])
            
            term_x = psi(flipped_spins_x, 1) / psi(spins, 1)
            term_y = psi(flipped_spins_y, 1) / psi(spins, 1)
            term_z = spins[:, i] * spins[:, (i + 1) % N]
            
            return term_x + term_y + term_z

        interaction_term = jnp.sum(jax.vmap(interaction_term_component, in_axes=(0, None))(jnp.arange(N), spins), axis=0)

        hamiltonian_operator = interaction_term

        if self.first_pass:
            self.energy = Energies.heisenberg(N)
            self.first_pass = False

        hamiltonian_operator = jnp.expand_dims(hamiltonian_operator, axis=0)

        return hamiltonian_operator