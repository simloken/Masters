import tensorflow as tf
import numpy as np
from energies import Energies

class NN:
    def __init__(self, hamiltonian, params):
        self.hamiltonian = hamiltonian
        self.params = params
        self.first_pass = True
        self.x_0 = False
        
        accepted_hamiltonians = ['two_fermions', 'calogero_sutherland',
                                 'ising', 'heisenberg']
        
        if hamiltonian not in accepted_hamiltonians:
            raise ValueError('Unrecognized Hamiltonian, try: ', accepted_hamiltonians)
        
        if hamiltonian == 'two_fermions':
            self.hamiltonian = self.two_fermions
            self.name = 'two_fermions'
            self.has_plots = True
            self.spin = False
        elif hamiltonian == 'calogero_sutherland':
            self.hamiltonian = self.calogero_sutherland
            self.x_0 = 0.5
            self.name = 'calogero_sutherland'
            self.has_plots = True
            self.spin = False
        elif hamiltonian == 'ising':
            self.hamiltonian = self.ising
            self.name = 'ising'
            self.has_plots = False
            self.spin = True
        elif hamiltonian == 'heisenberg':
            self.hamiltonian = self.heisenberg
            self.name = 'heisenberg'
            self.has_plots = False
            self.spin = True


        
        
    def two_fermions(self, psi, positions):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            psi (callable): A TensorFlow neural network estimating the wavefunction.
            positions (list of tf.Tensor): List of positional tensors for each particle.
            
        Returns:
            tf.Tensor: The Hamiltonian operator.
        """
                
        x1 = positions[0]
        x2 = positions[1]
        
        N, M = x1.shape
        
        omega = self.params

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x1)
            tape.watch(x2)
            psi_x1 = psi(x1)
            psi_x2 = psi(x2)
            
        gradient_psi_x1 = tape.gradient(psi_x1, x1)
        gradient_psi_x2 = tape.gradient(psi_x2, x2)
        
        kinetic_energy_1 = -0.5 * tf.reduce_sum(tf.square(gradient_psi_x1), axis=1)
        kinetic_energy_2 = -0.5 * tf.reduce_sum(tf.square(gradient_psi_x2), axis=1)

        del tape

        potential_energy_1 = 0.5 * omega**2 * tf.reduce_sum(tf.square(x1), axis=1)
        potential_energy_2 = 0.5 * omega**2 * tf.reduce_sum(tf.square(x2), axis=1)

        epsilon = 1e-8
        r_ij = tf.norm(x1[:, tf.newaxis, :] - x2, axis=2)
        interaction_energy = tf.reduce_sum(1.0 / (r_ij + epsilon), axis=1)/N
        # interaction_energy = 0 #for no interaction

        hamiltonian_operator = kinetic_energy_1 + kinetic_energy_2 + potential_energy_1 + potential_energy_2 + interaction_energy
        
        # print('T: ', np.mean(kinetic_energy_1 + kinetic_energy_2))
        # print('V: ', np.mean(potential_energy_1 + potential_energy_2))
        # print('I: ', np.mean(interaction_energy))
        
        
        if self.first_pass:
            self.energy = Energies.two_fermions()
            self.first_pass = False
        
        return hamiltonian_operator
    
    
    def calogero_sutherland(self, psi, positions):
        """
        Calculate the Hamiltonian operator for the Calogero-Sutherland model.
    
        Args:
            psi (callable): A TensorFlow neural network estimating the wavefunction.
            positions (list of tf.Tensor): List of positional tensors for each particle.
    
        Returns:
            tf.Tensor: The Hamiltonian operator.
        """
        N = len(positions)
        
        
        omega, beta = self.params
        
        x_0 = self.x_0
    
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(positions)
            psi_values = [psi(p) for p in positions]

        kinetic_terms = [-0.5 * tf.reduce_sum(tf.square(tape.gradient(psi_i, p)), axis=1) for psi_i, p in zip(psi_values, positions)]
        potential_terms = [0.5 * omega**2 * tf.reduce_sum(tf.square(p), axis=1) for p in positions]
    
        interaction_energy = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                r_ij = tf.norm(positions[i] - positions[j], axis=1)
                modified_potential = tf.square(tf.nn.tanh(r_ij / x_0)) / tf.square(r_ij)
                interaction_energy += (beta * (beta - 1)) * modified_potential
                
        hamiltonian_operator = sum(kinetic_terms) + sum(potential_terms) + interaction_energy
              
        # print('T: ', np.mean(sum(kinetic_terms)))
        # print('V: ', np.mean(sum(potential_terms)))
        # print('I: ', np.mean(interaction_energy))
        
        if self.first_pass:
            self.energy = Energies.calogero_sutherland(N, omega, beta)
            self.first_pass = False
        
        return hamiltonian_operator
        
    def ising(self, psi, spins):
        """
        Calculate the Hamiltonian operator for the 1D Transverse Ising Model.

        Args:
            psi (callable): A TensorFlow neural network estimating the wavefunction.
            spins (list of tf.Tensor): List of spin configurations for each particle.

        Returns:
            tf.Tensor: The Hamiltonian operator.
        """
        L = len(spins)
        Gamma, V = self.params
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(spins)
            psi_values = [psi(spin) for spin in spins]

        sigma_x_terms = [Gamma * tf.reduce_sum(tf.math.real(tape.gradient(psi_i, spin)), axis=1) for psi_i, spin in zip(psi_values, spins)]

        sigma_z_terms = [V * tf.reduce_sum(tf.math.real(tf.math.conj(psi_i) * psi_j), axis=1) for psi_i, psi_j in zip(psi_values[:-1], psi_values[1:])]

        hamiltonian_operator = sum(sigma_x_terms) + sum(sigma_z_terms)

        if self.first_pass:
            self.energy = Energies.ising(L)
            self.first_pass = False

        return hamiltonian_operator
    
    def heisenberg(self, psi, positions):
        ...
    
    
class RBM:
    def __init__(self, hamiltonian, params):
        self.hamiltonian = hamiltonian
        self.params = params
        self.first_pass = True
        self.x_0 = False
        
        accepted_hamiltonians = ['two_fermions', 'calogero_sutherland']
        
        if hamiltonian not in accepted_hamiltonians:
            raise ValueError('Unrecognized Hamiltonian, try: \n', accepted_hamiltonians)
        
        if hamiltonian == 'two_fermions':
            self.hamiltonian = self.two_fermions
        elif hamiltonian == 'calogero_sutherland':
            self.hamiltonian = self.calogero_sutherland
            self.x_0 = 0.5
            
    def two_fermions(self, r, dof):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            r (np.ndarray): The positions of particles in the system.
            dof (int): The degrees of freedom of the system

        Returns:
            float: The Hamiltonian operator
        """
        N = len(r) // dof
        
        omega = self.params
        
        kinetic_energy = -0.5 * np.sum(np.gradient(np.gradient(r)))
        potential_energy =  0.5 * omega**2 * np.sum(r**2)
        
        interaction_energy = 0

        r_reshaped = r.reshape(N, dof)
        delta_r = r_reshaped[:, np.newaxis, :] - r_reshaped[np.newaxis, :, :]
        distances = np.linalg.norm(delta_r, axis=2)
        interaction_energy = np.sum(1.0 / distances[np.triu_indices(N, k=1)])
        
        if self.first_pass:
            self.energy = Energies.two_fermions()
            self.first_pass = False
        
        return kinetic_energy + potential_energy + interaction_energy
    
    def calogero_sutherland(self, r, dof):
        """
        Calculate the Hamiltonian for the Calogero-Sutherland model.
         
        Args:
            r (np.ndarray) The positions of particles in the system.
            dof (int): The degrees of freedom of the system. Should always be 1.
         
        Returns:
            float: The Hamiltonian operator
        """
        
        omega = self.params[0]
        beta = self.params[1]
        
        x_0 = self.x_0
                      
        
        r_ij = np.abs(np.subtract.outer(r, r))
        
        r_ij = r_ij + 1e-8*np.identity(len(r))
        
        regularization = np.tanh(r_ij / x_0) ** 2 / r_ij
        
        kinetic_energy = -0.5 * np.sum(np.gradient(np.gradient(r)))
        potential_energy = 0.5 * omega * np.sum(r**2) + beta * (beta - 1) * np.sum(regularization)
        
        total_energy = kinetic_energy + potential_energy
        
        if self.first_pass:
            self.energy = Energies.calogero_sutherland(len(r), omega, beta)
            self.first_pass = False
        
        return total_energy
    
    def ising():
        ...
        
    def heisenberg():
        ...