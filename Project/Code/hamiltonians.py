import tensorflow as tf
import numpy as np

class NN:
    def two_fermions(psi, x1, x2, omega=1):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            psi (callable): A TensorFlow neural network estimating the wavefunction.
            x1 (tf.Tensor): Positional tensor for electron 1 with shape (N, M).
            x2 (tf.Tensor): Positional tensor for electron 2 with shape (N, M).
            omega (float, optional): The harmonic oscillator frequency.

        Returns:
            tf.Tensor: The Hamiltonian operator.
        """
        N, M = x1.shape

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

        return hamiltonian_operator
    
    
class RBM:
    def two_fermions(r, omega, dof):
        """
        Calculate the Hamiltonian operator for two interacting fermions (electrons).

        Args:
            r (np.ndarray): The positions of particles in the system.
            omega (float): The harmonic oscillator frequency.
            dof (int): The degrees of freedom of the system

        Returns:
            float: The Hamiltonian operator
        """
        N = len(r) // dof
        
        kinetic_energy = -0.5 * np.sum(np.gradient(np.gradient(r)))
        potential_energy =  0.5 * omega**2 * np.sum(r**2)
        
        interaction_energy = 0

        r_reshaped = r.reshape(N, dof)
        delta_r = r_reshaped[:, np.newaxis, :] - r_reshaped[np.newaxis, :, :]
        distances = np.linalg.norm(delta_r, axis=2)
        interaction_energy = np.sum(1.0 / distances[np.triu_indices(N, k=1)])
        
        return kinetic_energy + potential_energy + interaction_energy
    
    def calogero_sutherland(r, beta, dof=1):
        """
        Calculate the Hamiltonian for the Calogero-Sutherland model.
         
        Args:
            r (np.ndarray) The positions of particles in the system.
            beta (float): Interaction parameter
            dof (int): The degrees of freedom of the system. Should always be 1.
         
        Returns:
            float: The Hamiltonian operator
        """
        N = len(r)
        x0 = 0.5
        kinetic_energy = -0.5 * np.sum(np.diff(r, 2)**2)
        potential_energy = 0.5 * np.sum(r**2)
        
        distances = np.abs(r[:, None] - r[None, :])
        np.fill_diagonal(distances, 1.0)
        interaction_energy = np.sum((beta * (beta - 1)) * (np.tanh(distances / x0)**2) / distances**2)
        
        return kinetic_energy + potential_energy + interaction_energy