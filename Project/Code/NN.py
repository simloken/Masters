import tensorflow as tf
import numpy as np
    
def local_energy(psi, H, positions):
    """
    Calculate the local energy for the ground state.

    Args:
        psi (callable): A TensorFlow neural network estimating the wavefunction.
        H (tf.Tensor): The hamiltonian operator of the wavefunction.
        positions (list of tf.Tensor): List of positional tensors for each particle.

    Returns:
        tf.Tensor: The local energy for the ground state.
    """
    psi_vals = [psi(x) for x in positions]
    psi_product = tf.reduce_prod(psi_vals, axis=0)

    local_energy = tf.reduce_mean(H / psi_product)

    return local_energy

class NeuralNetwork(tf.Module):
    """
    A neural network model for variational quantum Monte Carlo (VMC) calculations.

    This class defines a feedforward neural network with multiple hidden layers
    for representing the wavefunction in VMC calculations.
    
    Methods:
        __call__(x):
            Compute the forward pass of the neural network given input `x`.
    """
    def __init__(self, l2_regularization=0.1, dropout_rate=0.3):
        self.layer1 = tf.keras.layers.Dense(256, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer2 = tf.keras.layers.Dense(128, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer3 = tf.keras.layers.Dense(64, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer4 = tf.keras.layers.Dense(16, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.layer5 = tf.keras.layers.Dense(4, activation='tanh',
                                            kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.output = tf.keras.layers.Dense(1, activation='tanh')

    def __call__(self, x):
        """
        Compute the forward pass of the neural network.

        Args:
            x (tf.Tensor): Input tensor.

        Returns:
        tf.Tensor: Output tensor representing the neural network's prediction.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        x = self.layer5(x)
        x = self.output(x)
        return x



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
        get_trainable_variables():
            Get the trainable variables of the wavefunction model.
    """
    def __init__(self):
        self.NN = NeuralNetwork()

    def __call__(self, x):
        """
        Compute the wavefunction for a given input configuration.

        Args:
            x (tf.Tensor): Input configuration tensor.

        Returns:
            tf.Tensor: Wavefunction values for the input configuration.
        """
        return self.NN(x)

    def get_trainable_variables(self):
        """
        Get the trainable variables of the wavefunction model.

        Returns:
            List[tf.Variable]: List of trainable variables.
        """
        return self.NN.trainable_variables

    
def print_trainable_variables(wavefunction):
    """
    Print the names and shapes of trainable variables in a given wavefunction.

    Args:
        wavefunction (WaveFunction): The wavefunction object containing trainable variables.

    Returns:
        None
    """
    trainable_variables = wavefunction.get_trainable_variables()
    for var in trainable_variables:
        print(f"Variable Name: {var.name}, Variable Shape: {var.shape}")

def metropolis_hastings_update(x, psi, delta):
    """
    Perform a Metropolis-Hastings update for a set of configurations.

    Args:
        x (tf.Tensor): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.
        delta (float): The step size for the Metropolis-Hastings update.

    Returns:
        tf.Tensor: Updated configurations after Metropolis-Hastings updates, shape (num_samples, dof).
    """
    num_samples, dof = x.shape
    
    proposed_x = x + delta * tf.random.normal(x.shape, dtype=tf.float32)
    
    psi_current = psi(x)
    psi_proposed = psi(proposed_x)
    
    acceptance_prob = tf.minimum(1.0, tf.square(tf.abs(psi_proposed) / tf.abs(psi_current)))
    random_numbers = tf.random.uniform((num_samples,))
    accept_mask = random_numbers[:, tf.newaxis] < acceptance_prob
    
    x = tf.where(accept_mask, proposed_x, x)
    
    return x


def variational_monte_carlo(wavefunction, hamiltonian, num_particles, num_samples, num_iterations, 
                            learning_rate, dof, delta, firstrun=True, target_energy=None, verbose=None):
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
        firstrun (bool): Flag to indicate if this is the first run (for handling NaN/inf values).
        target_energy (float): The target energy (if known).
        verbose (bool): Whether or not to print progress.

    Returns:
        energy (float): The final ground state energy of the system
    """
        
    optimizer = tf.optimizers.Adam(learning_rate)
    
    positions = [tf.Variable(tf.random.normal((num_samples, dof), dtype=tf.float32), trainable=True) 
                 for _ in range(num_particles)]

    initial_learning_rate = learning_rate
    
    energies = []
    

    for iteration in range(num_iterations):
        for i in range(num_particles):
            positions[i] = metropolis_hastings_update(positions[i], wavefunction, delta)

        with tf.GradientTape(persistent=True) as tape:
            energy = hamiltonian(wavefunction, positions)
            
            
        gradients = tape.gradient(energy, wavefunction.get_trainable_variables())

        energy = local_energy(wavefunction, energy, positions)
        
        energy_difference = 0
        if target_energy is not None:
            energy_difference = abs(target_energy - energy)
    
        
        if np.isnan(energy.numpy()).any() or np.isinf(energy.numpy()).any():
            if firstrun and verbose:
                print("NaN or inf encountered in configuration. Trying new initial configuration(s)...")
            wavefunction = WaveFunction()
            return variational_monte_carlo(wavefunction, hamiltonian, num_particles, 
                                           num_samples, num_iterations,
                                           learning_rate, dof, delta, firstrun=False,
                                           target_energy=target_energy, verbose=verbose)
        
        
        updated_learning_rate = initial_learning_rate * (1.0 + 0.1 * energy_difference)


        optimizer.apply_gradients(zip(gradients, wavefunction.get_trainable_variables()))
        
        learning_rate = updated_learning_rate
        
    
        if iteration % 10 == 0 and verbose:
            print(f"Iteration {int(iteration/10+1)}: Energy = {energy.numpy():.3f} a.u.")
        
        energies.append(energy.numpy())
    
    return energy.numpy()