import tensorflow as tf

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
                
            
        self.kinetic_energy = []
        self.potential_energy = [] 
    def harmonic_oscillator(self, psi, positions):
         """
         Calculate the Hamiltonian operator for a 1D harmonic oscillator.
     
         Args:
             psi (callable): A TensorFlow neural network estimating the wavefunction.
             positions (list of tf.Tensor): List of positional tensors for each particle.
             
         Returns:
             tf.Tensor: The Hamiltonian operator.
         """
         
         
         x = positions
                 
         omega = self.params
     
         with tf.GradientTape() as tape:
             tape.watch(x)
             psi_x = psi(x)
             
         gradient_psi_x = tape.gradient(psi_x, x)
         
         kinetic_energy = -0.5 * tf.reduce_sum(tf.square(gradient_psi_x), axis=1)
         
         self.kinetic_energy.append(kinetic_energy[0])
         
         potential_energy = 0.5 * omega**2 * tf.reduce_sum(tf.square(x), axis=1) * tf.squeeze(psi_x)
                 
         self.potential_energy.append(potential_energy[0])
     
         hamiltonian_operator = kinetic_energy + potential_energy
     
         if self.first_pass:
             self.energy = Energies.harmonic_oscillator(omega)
             self.first_pass = False
             
         hamiltonian_operator = tf.expand_dims(hamiltonian_operator, axis=0)
         
         
         return hamiltonian_operator
     
     
         
    def two_fermions(self, psi, positions):
         """
         Calculate the Hamiltonian operator for two interacting fermions (electrons).
     
         Args:
             psi (callable): A TensorFlow neural network estimating the wavefunction.
             positions (list of tf.Tensor): List of positional tensors for each particle.
             
         Returns:
             tf.Tensor: The Hamiltonian operator.
         """
                      
         x = positions
         
         N, dof_times_particles = x.shape
         
         
         x1, x2 = tf.split(x, num_or_size_splits=2, axis=1)
         
         omega = self.params
     
         with tf.GradientTape(persistent=True) as tape:
             tape.watch(x)
             psi_x = psi(x)
          
         gradient_psi_x = tape.gradient(psi_x, x)
         gradient_psi_x_i = tf.split(gradient_psi_x, num_or_size_splits=2, axis=1)
         
         kinetic_energy_1 = -0.5 * tf.reduce_sum(tf.square(gradient_psi_x_i[0]), axis=1)        
         kinetic_energy_2 = -0.5 * tf.reduce_sum(tf.square(gradient_psi_x_i[1]), axis=1)
         del tape
         potential_energy_1 = 0.5 * omega**2 * tf.reduce_sum(tf.square(x1), axis=1) * tf.squeeze(psi_x)
         potential_energy_2 = 0.5 * omega**2 * tf.reduce_sum(tf.square(x2), axis=1) * tf.squeeze(psi_x)
         
         epsilon = 1e-8
         r_ij = tf.norm(x1[:, tf.newaxis, :] - x2, axis=2)
         interaction_energy = tf.squeeze(psi_x) * tf.reduce_sum(1.0 / (r_ij + epsilon), axis=1)/N
     
         hamiltonian_operator = kinetic_energy_1 + kinetic_energy_2 + potential_energy_1 + potential_energy_2 + interaction_energy
     
         
         
         if self.first_pass:
             self.energy = Energies.two_fermions()
             self.first_pass = False
             
         hamiltonian_operator = tf.expand_dims(hamiltonian_operator, axis=0)
         
     
         
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
         samples, N = np.shape(positions)
         
         omega, beta = self.params
         
         x_0 = self.x_0
     
         hamiltonian_operator = 0
         
         x = positions
     
     
         with tf.GradientTape() as tape:
             tape.watch(x)
             psi_x = psi(x)
                 
         gradient_psi_x = tape.gradient(psi_x, x)
         gradient_psi_x_i = tf.split(gradient_psi_x, num_or_size_splits=N, axis=1)
         kinetic_energy = 0
         potential_energy = 0
         for i in range(N):
             kinetic_energy += tf.reduce_sum(-0.5 * tf.reduce_sum(tf.square(gradient_psi_x_i[i]), axis=1))
             
         potential_energy = 0.5 * omega**2 * tf.reduce_sum(tf.square(x), axis=1) * tf.squeeze(psi_x)
         hamiltonian_operator = kinetic_energy + potential_energy
     
         x_expanded_1 = tf.expand_dims(x, axis=2)  # Shape (num_samples, num_particles, 1)
         x_expanded_2 = tf.expand_dims(x, axis=1)  # Shape (num_samples, 1, num_particles)
     
         # Compute pairwise differences
         x_ij = x_expanded_1 - x_expanded_2  # Shape (num_samples, num_particles, num_particles)
     
         # Compute the interaction term using the formula
         # Adding epsilon to the denominator to avoid division by zero
         term = beta * (beta - 1) * tf.square(tf.tanh(x_ij / x_0)) / (tf.square(x_ij) + 1e-8)
     
         # Create a mask to ignore the diagonal (i == j)
         mask = tf.linalg.band_part(tf.ones((N, N)), 0, -1) - tf.eye(N)
     
         # Apply the mask to set the diagonal terms to zero
         term = term * mask
     
         # Sum over the pairwise interactions for each sample
         interaction_energy = tf.reduce_sum(term, axis=[1, 2])
                 
         hamiltonian_operator += interaction_energy * tf.squeeze(psi_x)
                     
         
         if self.first_pass:
             self.energy = Energies.calogero_sutherland(N, omega, beta)
             self.first_pass = False
             
             
         
         hamiltonian_operator = tf.expand_dims(hamiltonian_operator, axis=0)
         
                     
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
             
         Gamma, V = self.params
     
         samples, N = spins.shape
             
         # Calculate the interaction term
         interaction_term = V * tf.reduce_sum(spins * tf.roll(spins, -1, axis=1), axis=1)
         
         # Calculate the transverse field term
         transverse_field_term = 0
         for i in range(N):
             flipped_spins = tf.identity(spins)
             flipped_spins = flipped_spins.numpy()
             flipped_spins[:, i] *= -1
             transverse_field_term += Gamma * (psi(tf.convert_to_tensor(flipped_spins)) / psi(spins))
             
         hamiltonian_operator = interaction_term + transverse_field_term
      
         if self.first_pass:
             self.energy = Energies.ising(N)
             self.first_pass = False
                             
         hamiltonian_operator = tf.expand_dims(hamiltonian_operator, axis=0)
     
         return hamiltonian_operator
     
    def heisenberg(self, psi, spins):
         """
         Calculate the Hamiltonian operator for the 1/2 spin Heisenberg antiferromagnetic chain.
     
         Args:
             psi (callable): A TensorFlow neural network estimating the wavefunction.
             spins (list of tf.Tensor): List of spin configurations for each particle.
     
         Returns:
             tf.Tensor: The Hamiltonian operator.
         """
         
         samples, N = spins.shape
                 
         interaction_term = 0
         for i in range(N):
             # S^x_i S^x_{i+1}
             flipped_spins_x = tf.identity(spins)
             flipped_spins_x = flipped_spins_x.numpy()
             flipped_spins_x[:, i] *= -1
             flipped_spins_x[:, (i+1) % N] *= -1
             
             # S^y_i S^y_{i+1}
             flipped_spins_y = tf.cast(tf.identity(spins), tf.complex64)
             flipped_spins_y = flipped_spins_y.numpy()
             flipped_spins_y[:, i] *= -1j
             flipped_spins_y[:, (i+1) % N] *= 1j
             
             interaction_term += (
                 (psi(tf.convert_to_tensor(flipped_spins_x)) / psi(spins)) + 
                 (psi(tf.convert_to_tensor(flipped_spins_y)) / psi(spins)) +
                 (spins[:, i] * spins[:, (i+1) % N])
             )
         
         hamiltonian_operator = interaction_term
         
         if self.first_pass:
             self.energy = Energies.heisenberg(N)
             self.first_pass = False
                             
         hamiltonian_operator = tf.expand_dims(hamiltonian_operator, axis=0)
     
         return hamiltonian_operator