import tensorflow as tf
import numpy as np
   
def loss(wavefunction, model, samples):
    """
    Calculate the loss function for Variational Monte Carlo (VMC) optimization.

    This function computes the loss used during VMC optimization to estimate the ground state energy of a quantum system.

    Args:
        wavefunction (callable): A TensorFlow neural network estimating the wavefunction.
        model (object): The Hamiltonian object
        samples (list of tf.Tensor): List of positional tensors for each particle.

    Returns:
        tf.Tensor: The energy loss value to be minimized during VMC optimization.
    """
    
    if model.name == 'calogero_sutherland': #must be sorted as they are bosons
        samples = tf.sort(samples, axis=1)
       
    H_psi = model.hamiltonian(wavefunction, samples)
    

    psi_vals = wavefunction(samples)
    psi_star_vals = tf.math.conj(wavefunction(samples))
    psi_magnitude_squared = tf.reduce_sum(tf.square(psi_vals))/len(samples) #why does it sum to ~len(samples) and not ~1???
    
    expectation_H = tf.multiply(psi_star_vals, H_psi)
    loss_value = tf.reduce_mean(expectation_H / psi_magnitude_squared)
        
    return loss_value


def normalize(wavefunction, samples, name):
    """
    Normalize the wavefunction using Monte Carlo estimation.

    Args:
        wavefunction (callable): A TensorFlow neural network estimating the wavefunction.
        samples (list of tf.Tensor): List of positional tensors for each particle.

    Returns:
        callable: The normalized wavefunction.
    """
    if name == 'calogero_sutherland':
        samples = tf.sort(samples, axis=0)
    
    # Compute the wavefunction values at the sample points
    psi_vals = wavefunction(samples)
    # Compute the square of the magnitudes of the wavefunction values
    psi_magnitude_squared = tf.square(psi_vals)
    
    # Estimate the normalization constant using Monte Carlo integration
    integral = tf.reduce_mean(psi_magnitude_squared)
    
    # Define the normalized wavefunction
    def normalized_wavefunction(x):
        return wavefunction(x) / tf.sqrt(integral)
    
    return normalized_wavefunction


class NeuralNetwork(tf.Module):
    """
    A neural network model for variational quantum Monte Carlo (VMC) calculations.

    This class defines a feedforward neural network with multiple hidden layers
    for representing the wavefunction in VMC calculations.
    
    Methods:
        __call__(x):
            Compute the forward pass of the neural network given input `x`.
    """
    def __init__(self, dof, regularization=.3):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='softplus', kernel_regularizer=tf.keras.regularizers.l2(regularization), input_shape=(dof,)),
            tf.keras.layers.Dense(32, activation='softplus', kernel_regularizer=tf.keras.regularizers.l2(regularization)),
            tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(regularization))
        ])

    def __call__(self, x):
        """
        Compute the forward pass of the neural network.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
        tf.Tensor: Output tensor representing the neural network's prediction.
        """
        return tf.abs(self.model(x))



class WaveFunction(tf.Module):
    """
    Wavefunction model based on a neural network.

    This class encapsulates the wavefunction model used in variational quantum
    Monte Carlo (VMC) calculations. It utilizes a neural network defined by the
    NeuralNetwork class.

    Attributes:
        NN: A neural network model (NeuralNetwork instance).

    Methods:
        __call__(x):
            Compute the wavefunction for a given input configuration `x`.
    """
    def __init__(self, particles, hamiltonian):
        self.H_name = hamiltonian
        if self.H_name == 'two_fermions':
            dof = 2*particles
        elif self.H_name in ['ising', 'heisenberg']:
            dof = particles
        else:
            dof = particles
        self.NN = NeuralNetwork(dof=dof)
    def __call__(self, x):
        """
        Compute the wavefunction for a given input configuration.

        Args:
            x (tf.Tensor): Input configuration tensor.

        Returns:
            tf.Tensor: Wavefunction values for the input configuration.
        """
        # print(tf.reduce_mean(self.NN(x)).numpy())
        return self.NN(x)



def compute_force(x, psi):
    """
    Compute the quantum force for a given configuration.

    Args:
        x (tf.Tensor): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.

    Returns:
        tf.Tensor: The computed quantum force, shape (num_samples, dof).
    """
    with tf.GradientTape() as tape:
        tape.watch(x)
        psi_x = psi(x)
    grad_psi = tape.gradient(psi_x, x)
    force = 2 * grad_psi / psi_x
    return force

def compute_greens_function(x, proposed_x, F, proposed_F, delta):
    """
    Compute the Green's function for a given configuration.

    Args:
        x (tf.Tensor): The old configurations, shape (num_samples, dof).
        proposed_x (tf.Tensor): The new configurations, shape (num_samples, dof).
        F (tf.Tensor): The old quantum force, shape (num_samples, dof).
        proposed_F (tf.Tensor): The new quantum force, shape (num_samples, dof).

    Returns:
        tf.Tensor: The computed Green's function, shape (num_samples,).
    """
    G = 0.5 * (F + proposed_F) * (0.5**2 * delta * (F - proposed_F) - proposed_x + x)
    G = tf.reduce_sum(G, axis=1)
    G = tf.exp(G)
    return G

def metropolis_hastings_update(x, num_particles, psi, delta, hamiltonian):
    """
    Perform a Metropolis-Hastings update for a set of configurations.

    Args:
        x (tf.Tensor): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.
        delta (float): The step size for the Metropolis-Hastings update.

    Returns:
        tf.Tensor: Updated configurations after Metropolis-Hastings updates, shape (num_samples, dof).
    """
    
    name = hamiltonian.name
    
    num_samples, dof_times_particles = np.shape(x)
    dof = int(dof_times_particles/num_particles)
    
    F = compute_force(x, psi)
    
    proposed_x = x + np.sqrt(delta) * tf.random.normal(x.shape) + 0.5 * delta * F
    proposed_F = compute_force(proposed_x, psi)
        
    if name == 'calogero_sutherland':
        x = tf.sort(x, axis=1)
        proposed_x = tf.sort(proposed_x, axis=1)
      
    psi_current = psi(x)
    psi_proposed = psi(proposed_x)
    
    
    G = compute_greens_function(x, proposed_x, F, proposed_F, delta)

    if dof == 2:
        G = tf.expand_dims(G, axis=-1)
        acceptance_prob = tf.minimum(1.0, G * tf.minimum(1.0, tf.square(psi_proposed / psi_current)))
    else:
        acceptance_prob = tf.minimum(1.0, tf.reshape(G * tf.squeeze(tf.minimum(1.0, tf.square(psi_proposed / psi_current))), (-1,1)))
        
    # print('Mean acceptance probability:', tf.reduce_mean(acceptance_prob).numpy())
    
    random_numbers = tf.random.uniform((num_samples,))
    accept_mask = random_numbers[:, tf.newaxis] < acceptance_prob
    x = tf.where(accept_mask, proposed_x, x)
    return x

def metropolis_hastings_spin_update(samples, psi, hamiltonian):
    """
    Perform a Metropolis-Hastings update for a set of spin configurations.

    Args:
        samples (tf.Tensor): The current spin configurations, shape (num_samples, num_particles, 1).
        psi (callable): A function that computes the wavefunction for a given configuration.

    Returns:
        tf.Tensor: Updated spin configurations after Metropolis-Hastings updates, shape (num_samples, num_particles, 1).
    """
    
    
    M, N = np.shape(samples)
    
    # Generate random indices for each sample
    rand_indices = tf.random.uniform(shape=(M,), minval=0, maxval=N, dtype=tf.int32)
    
    # Create indices for scatter update
    indices = tf.stack([rand_indices, tf.range(M)], axis=1)
    
    # Select the particles and flip their signs
    updates = -tf.gather_nd(samples, indices)
    
    # Apply the updates to create the proposed_samples array
    proposed_samples = tf.tensor_scatter_nd_update(samples, indices, updates)

    # Compute the wavefunctions for the current and proposed samples
    psi_current = psi(samples)
    psi_proposed = psi(proposed_samples)
    
    # Compute the acceptance probabilities
    acceptance_prob = tf.minimum(1.0, (psi_proposed / psi_current) ** 2)
    # Determine which configurations are accepted
    random_numbers = tf.random.uniform((M,))
    accept_mask = random_numbers[:, tf.newaxis] < acceptance_prob

    # Only update the accepted configurations
    samples = tf.where(accept_mask, proposed_samples, samples)
    
    
    return samples

def variational_monte_carlo(wavefunction, hamiltonian, num_particles, num_samples,
                            num_iterations, learning_rate, dof, delta,
                            verbose=None, debug=False):
    """
    Perform Variational Monte Carlo (VMC) optimization to find the ground state energy.

    Args:
        wavefunction (WaveFunction): The wavefunction to optimize.
        hamiltonian (function): The hamiltonian of the system
        num_particles (int): The number of particles in the system
        num_samples (int): The number of configurations to sample.
        num_iterations (int): The number of VMC iterations.
        learning_rate (float): The initial learning rate for the optimizer.
        dof (int): The degrees of freedom in the system.
        delta (float): The Metropolis-Hastings step size.
        verbose (bool): Whether or not to print progress.

    Returns:
        energy (float): The final ground state energy of the system
    """
    
    optimizer = tf.keras.optimizers.Adam(learning_rate, clipvalue=1.0)
    
    model = wavefunction.NN.model
                      
    
    
    if hamiltonian.spin: #lattice particle spin
        samples = [tf.where(tf.random.uniform((num_samples, dof)) > 0.5, 0.5, -0.5)
         for _ in range(num_particles)]

    else: #free particles
        samples = [tf.random.normal((num_samples, dof), stddev=3, dtype=tf.float32) 
                 for _ in range(num_particles)]
        
    samples = tf.concat(samples, axis=-1)
    
        
    energies = []
        
    for iteration in range(num_iterations):
        psi = normalize(wavefunction, samples, hamiltonian.name)

        if not hamiltonian.spin: #free particles
            samples = metropolis_hastings_update(samples, num_particles, psi, delta, hamiltonian)
                
        if hamiltonian.spin: #lattice spin
            samples = np.array(samples)
            psi = normalize(wavefunction, samples, hamiltonian.name)
            samples = metropolis_hastings_spin_update(samples, psi, hamiltonian)
            
        with tf.GradientTape(persistent=True) as tape:
            psi = normalize(wavefunction, samples, hamiltonian.name) #renormalize after moves (recalculating normalization constant)
            loss_value = loss(psi, hamiltonian, samples)

        
        gradients = tape.gradient(loss_value, model.trainable_variables)
                
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # for i, grad in enumerate(gradients):
        #     tf.debugging.check_numerics(grad, 'Invalid gradient at index {}'.format(i))
        
        if iteration % 10 == 0 and verbose:
            print(f'{iteration}: {loss_value.numpy()}')
        elif iteration+1 == num_iterations and verbose:
            print(f'{iteration+1}: {loss_value.numpy()}')
            
        energies.append(loss_value.numpy())
        
        if hamiltonian.x_0: # ugly
            factor = (hamiltonian.x_0_initial-hamiltonian.x_0_minimum)/num_iterations
            if hamiltonian.x_0 > hamiltonian.x_0_minimum:  
                hamiltonian.x_0 -= factor
                if hamiltonian.x_0 < hamiltonian.x_0_minimum:
                    hamiltonian.x_0 = hamiltonian.x_0_minimum
            else:
                hamiltonian.x_0 = hamiltonian.x_0_minimum

    samples = tf.concat(samples, axis=1).numpy()
            
    return loss_value.numpy(), hamiltonian, np.real(energies), samples
