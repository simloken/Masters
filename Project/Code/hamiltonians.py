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
            self.has_plots = True
            self.spin = False
            self.symmetric = True
            if self.params == 'default':
                self.params = 1
        
        elif hamiltonian == 'two_fermions':
            self.hamiltonian = self.two_fermions
            self.name = 'two_fermions'
            self.has_plots = True
            self.spin = False
            self.symmetric = False
            if self.params == 'default':
                self.params = 1
            
        elif hamiltonian == 'calogero_sutherland':
            self.hamiltonian = self.calogero_sutherland
            self.x_0 = 1
            self.x_0_initial = self.x_0
            self.x_0_minimum = 0.68
            self.name = 'calogero_sutherland'
            self.has_plots = True
            self.spin = False
            self.symmetric = '!'
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
        
        gradient_psi_x = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        
        gradient_psi_x2 = torch.autograd.grad(gradient_psi_x, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
                
        kinetic_energy = - 1/2 * gradient_psi_x2.sum(dim=1, keepdim=True)
        
        potential_energy = 0.5*omega**2 * x**2 * psi_x
        
        H_Psi = kinetic_energy + potential_energy

        if self.first_pass:
            self.energy = Energies.harmonic_oscillator(omega)
            self.first_pass = False
        
        # print('K:', torch.mean(kinetic_energy).detach().numpy())
        # print('V:', torch.mean(potential_energy).detach().numpy())
        
        return H_Psi

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
        
        gradient_psi_x = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
        laplacian_psi_x = torch.autograd.grad(gradient_psi_x, x, grad_outputs=torch.ones_like(gradient_psi_x), create_graph=True)[0]
        
        kinetic_energy = -0.5 * laplacian_psi_x.sum(dim=1, keepdim=True)
        
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
        
        x = positions
        omega, beta = self.params
        x_0 = self.x_0


        x.requires_grad_(True)
        N, M = x.shape
        psi_x = psi(x) 
        gradient_psi_x = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
        laplacian_psi_x = torch.autograd.grad(gradient_psi_x, x, grad_outputs=torch.ones_like(gradient_psi_x), create_graph=True)[0]
        
        kinetic_energy = -0.5 * laplacian_psi_x

        r_squared = x**2
        potential_energy = 0.5 * omega**2 * r_squared*psi_x
        
        interaction_energy = 0
        for i in range(M):
            for j in range(i + 1, M):
                x_ij = positions[:, i] - positions[:, j]
                interaction_energy += beta * (beta - 1) * (torch.tanh(x_ij / x_0)**2) / (x_ij**2)
        
        H_Psi = kinetic_energy.sum(axis=1) + potential_energy.sum(axis=1) + interaction_energy * psi_x
        

        if self.first_pass:
            self.energy = Energies.calogero_sutherland(M, omega, beta)
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
        N, M = spins.shape
        
        interaction_term = torch.zeros(N)
        for i in range(M):
            interaction_term -= V * spins[:, i] * spins[:, (i+1) % M] * psi(spins).squeeze()
        
        transverse_term = torch.zeros(N)
        for i in range(M):
            flipped_spins = spins.clone()
            flipped_spins[:, i] *= -1
            transverse_term -= Gamma * psi(flipped_spins).squeeze()
        
        H_Psi = interaction_term + transverse_term
                

        if self.first_pass:
            self.energy = Energies.ising(M, Gamma, V)
            self.first_pass = False

        return H_Psi

    def heisenberg(self, psi, spins):
        N, M = spins.shape
        J = 1
        H_Psi = torch.zeros(N)
        
        for i in range(M):
            next_i = (i + 1) % M
            
            flipped_spins_xx = spins.clone()
            flipped_spins_xx[:, i] *= -1
            flipped_spins_xx[:, next_i] *= -1
            
            flipped_spins_yy = spins.clone()
            flipped_spins_yy[:, i] *= -1
            flipped_spins_yy[:, next_i] *= -1
            
            sigma_z_term = spins[:, i] * spins[:, next_i]
            
            H_Psi += J * (psi(flipped_spins_xx).squeeze() + 
                          psi(flipped_spins_yy).squeeze() + 
                          sigma_z_term * psi(spins).squeeze())
    
        if self.first_pass:
            self.energy = Energies.heisenberg(M)
            self.first_pass = False
        
        return H_Psi
   
    
    
    
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
            self.symmetric = '!'
            self.binary = False
        
        elif hamiltonian == 'two_fermions':
            self.hamiltonian = self.two_fermions
            self.name = 'two_fermions'
            self.has_plots = False
            self.spin = False
            if self.params == 'default':
                self.params = 1
            self.symmetric = '!'
            self.binary = False
            
        elif hamiltonian == 'calogero_sutherland':
            self.hamiltonian = self.calogero_sutherland
            self.x_0 = 1
            self.x_0_initial = self.x_0
            self.x_0_minimum = 0.68
            self.name = 'calogero_sutherland'
            self.has_plots = True
            self.spin = False
            if self.params == 'default':
                self.params = [1,2]
            self.symmetric = True
            self.binary = False
            
        elif hamiltonian == 'ising':
            self.hamiltonian = self.ising
            self.name = 'ising'
            self.has_plots = False
            self.spin = True
            if self.params == 'default':
                self.params = [-1,-1]
            self.symmetric = ('!')
            self.binary = True
            
        elif hamiltonian == 'heisenberg':
            self.hamiltonian = self.heisenberg
            self.name = 'heisenberg'
            self.has_plots = False
            self.spin = True
            if self.params == 'default':
                self.params = []
            self.symmetric = ('!')
            self.binary = True
                
            
    def harmonic_oscillator(self, psi, positions, W, a, b):
        """
        Calculate the Hamiltonian operator for a 1D harmonic oscillator.

        Args:
            psi (callable): A function estimating the wavefunction.
            positions (jnp.ndarray): Positional tensors for each particle.
            W (jnp.ndarray): Weight matrix of the RBM.
            a (jnp.ndarray): Visible bias of the RBM.
            b (jnp.ndarray): Hidden bias of the RBM.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        x = positions
        omega = self.params
    
        psi_x = psi(x, 1.0, 0, W, a, b)
        
        gradient_psi_x2 = psi(x, 1, 2, W, a, b)
            
        kinetic_energy = -0.5 * gradient_psi_x2
        
        potential_energy = 0.5 * omega**2 * x.squeeze()**2 * psi_x.squeeze()
    
        H_Psi = kinetic_energy + potential_energy
    
        if self.first_pass:
            self.energy = Energies.harmonic_oscillator(omega)
            self.first_pass = False
                                
        return H_Psi
    
    
    def two_fermions(self, psi, positions, W, a, b):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            psi (callable): A function estimating the wavefunction.
            positions (jnp.ndarray): Positional tensors for each particle.
            W (jnp.ndarray): Weight matrix of the RBM.
            a (jnp.ndarray): Visible bias of the RBM.
            b (jnp.ndarray): Hidden bias of the RBM.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        x = positions
        omega = self.params
    
        psi_x = psi(x, 2, 0, W, a, b)
    
        gradient_psi_x2 = psi(x, 2, 2, W, a, b)
    
        kinetic_energy = -0.5 * gradient_psi_x2
        
        
    
        r_squared = jnp.sum(x**2, axis=1, keepdims=True)
        potential_energy = 0.5 * omega**2 * r_squared.squeeze() * psi_x
            
        r1 = x[:, :2]
        r2 = x[:, 2:]
        r12 = jnp.linalg.norm(r1 - r2, axis=1, keepdims=True)
        interaction_energy = (1 / r12) * psi_x
        H_psi = kinetic_energy + potential_energy + interaction_energy
    
        if self.first_pass:
            self.energy = Energies.two_fermions()
            self.first_pass = False

        return H_psi
    
    def calogero_sutherland(self, psi, positions, W, a, b):
        """
        Calculate the Hamiltonian operator for the Calogero-Sutherland model.

        Args:
            psi (callable): A function estimating the wavefunction.
            positions (jnp.ndarray): Positional tensors for each particle.
            W (jnp.ndarray): Weight matrix of the RBM.
            a (jnp.ndarray): Visible bias of the RBM.
            b (jnp.ndarray): Hidden bias of the RBM.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        x = positions
        omega, beta = self.params
        x_0 = self.x_0


        N, M = x.shape
        psi_x = psi(x, 1, 0, W, a, b)
    
        laplacian_psi_x = psi(x, 1, 2, W, a, b)
        
        kinetic_energy = -0.5 * laplacian_psi_x

        r_squared = x**2
        potential_energy = 0.5 * omega**2 * r_squared*jnp.expand_dims(psi_x, axis=1)
        1
        interaction_energy = 0
        for i in range(M):
            for j in range(i + 1, M):
                x_ij = positions[:, i] - positions[:, j]
                interaction_energy += beta * (beta - 1) * (jnp.tanh(x_ij / x_0)**2) / (x_ij**2)
        
        H_Psi = kinetic_energy + potential_energy.sum(axis=1) + interaction_energy * psi_x
        

        if self.first_pass:
            self.energy = Energies.calogero_sutherland(M, omega, beta)
            self.first_pass = False


        return H_Psi
    
    def ising(self, psi, spins, W, a, b):
        """
        Calculate the Hamiltonian operator for the Ising model.

        Args:
            psi (callable): A function estimating the wavefunction.
            spins (jnp.ndarray): Spin configurations.
            W (jnp.ndarray): Weight matrix of the RBM.
            a (jnp.ndarray): Visible bias of the RBM.
            b (jnp.ndarray): Hidden bias of the RBM.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        Gamma, V = self.params
        N, M = spins.shape
        
        interaction_term = jnp.zeros(N)
        for i in range(M):
            interaction_term -= V * spins[:, i] * spins[:, (i+1) % M] * psi(spins, 1, 0, W, a, b).squeeze()
        
        transverse_term = jnp.zeros(N)
        for i in range(M):
            flipped_spins = spins.clone()
            flipped_spins = flipped_spins.at[:, i].set(flipped_spins[:, i]*-1)
            transverse_term -= Gamma * psi(flipped_spins, 1, 0, W, a, b).squeeze()
        
        H_Psi = interaction_term + transverse_term
                

        if self.first_pass:
            self.energy = Energies.ising(M, Gamma, V)
            self.first_pass = False

        return H_Psi
    
    def heisenberg(self, psi, spins, W, a, b):
        """
        Calculate the Hamiltonian operator for the Heisenberg model.

        Args:
            psi (callable): A function estimating the wavefunction.
            spins (jnp.ndarray): Spin configurations.
            W (jnp.ndarray): Weight matrix of the RBM.
            a (jnp.ndarray): Visible bias of the RBM.
            b (jnp.ndarray): Hidden bias of the RBM.

        Returns:
            jnp.ndarray: The Hamiltonian operator.
        """
        N, M = spins.shape
        J = 1
        H_Psi = jnp.zeros(N)
        
        for i in range(M):
            next_i = (i + 1) % M
            
            flipped_spins_xx = spins.clone()
            flipped_spins_xx = flipped_spins_xx.at[:, i].set(flipped_spins_xx[:, i]*-1)
            flipped_spins_xx = flipped_spins_xx.at[:, next_i].set(flipped_spins_xx[:, i]*-1)
            
            flipped_spins_yy = spins.clone()
            flipped_spins_yy = flipped_spins_yy.at[:, i].set(flipped_spins_yy[:, i]*-1)
            flipped_spins_yy = flipped_spins_yy.at[:, next_i].set(flipped_spins_yy[:, i]*-1)
            
            sigma_z_term = spins[:, i] * spins[:, next_i]
            
            H_Psi += J * (psi(flipped_spins_xx, 1, 0, W, a, b).squeeze() + 
                          psi(flipped_spins_yy, 1, 0, W, a, b).squeeze() + 
                          sigma_z_term * psi(spins, 1, 0, W, a, b).squeeze())

        if self.first_pass:
            self.energy = Energies.heisenberg(M)
            self.first_pass = False

        
        return H_Psi