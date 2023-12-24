import tensorflow as tf
import numpy as np
    
def loss(wavefunction, model, positions):
    """
    Calculate the loss function for Variational Monte Carlo (VMC) optimization.

    This function computes the loss used during VMC optimization to estimate the ground state energy of a quantum system. The loss is defined as the negative of the energy expectation value, which is the ratio of the Hamiltonian expectation value to the square of the wavefunction magnitude.

    Args:
        wavefunction (callable): A TensorFlow neural network estimating the wavefunction.
        model (object): The Hamiltonian object
        positions (list of tf.Tensor): List of positional tensors for each particle.

    Returns:
        tf.Tensor: The energy loss value to be minimized during VMC optimization.
    """
    energy = model.hamiltonian(wavefunction, positions)
    
    psi_vals = [wavefunction(x) for x in positions]
    
    psi_product = tf.reduce_prod(psi_vals, axis=0)
    
    loss_value = tf.reduce_mean(energy / psi_product)
    
    return -loss_value


class NeuralNetwork(tf.Module):
    """
    A neural network model for variational quantum Monte Carlo (VMC) calculations.

    This class defines a feedforward neural network with multiple hidden layers
    for representing the wavefunction in VMC calculations.
    
    Methods:
        __call__(x):
            Compute the forward pass of the neural network given input `x`.
    """
    def __init__(self, l2_regularization=0.15, dropout_rate=0.3):
        self.layer1 = tf.keras.layers.Dense(512, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer2 = tf.keras.layers.Dense(256, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer3 = tf.keras.layers.Dense(128, activation='sigmoid',
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

def metropolis_hastings_spin_update(spins, psi, delta):
    """
    Perform a Metropolis-Hastings update for a set of spin configurations.

    Args:
        spins (tf.Tensor): The current spin configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.
        delta (float): The step size for the Metropolis-Hastings update.

    Returns:
        tf.Tensor: Updated spin configurations after Metropolis-Hastings updates, shape (num_samples, dof).
    """
    num_samples, dof = spins.shape
    
    proposed_spins = tf.where(tf.random.uniform(spins.shape) > 0.5, 0.5, -0.5)
    
    psi_current = psi(spins)
    psi_proposed = psi(proposed_spins)
    
    acceptance_prob = tf.minimum(1.0, tf.square(tf.abs(psi_proposed) / tf.abs(psi_current)))
    random_numbers = tf.random.uniform((num_samples,))
    accept_mask = random_numbers[:, tf.newaxis] < acceptance_prob
    
    spins = tf.where(accept_mask, proposed_spins, spins)
    
    return spins



def variational_monte_carlo(wavefunction, hamiltonian, num_particles, num_samples,
                            num_iterations, learning_rate, dof, delta,
                            firstrun=True, target_energy=None, verbose=None):
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
    
    if hamiltonian.spin: #lattice particle spin
        positions = [tf.Variable(tf.where(tf.random.uniform((num_samples, dof)) > 0.5, 0.5, -0.5), trainable=True) 
         for _ in range(num_particles)]

    else: #free particles
        positions = [tf.Variable(tf.random.normal((num_samples, dof), dtype=tf.float32), trainable=True) 
                 for _ in range(num_particles)]
        
    energies = []
        
    for iteration in range(num_iterations):
        for i in range(num_particles):
            if hamiltonian.spin:
                positions[i] = metropolis_hastings_spin_update(positions[i], wavefunction, delta)
            else:
                positions[i] = metropolis_hastings_update(positions[i], wavefunction, delta)

        with tf.GradientTape(persistent=True) as tape:
            loss_value = loss(wavefunction, hamiltonian, positions)

        gradients = tape.gradient(loss_value, wavefunction.get_trainable_variables())
        
        if np.isnan(loss_value.numpy()).any() or np.isinf(loss_value.numpy()).any():
            if firstrun and verbose:
                print("NaN or inf encountered in configuration. Trying new initial configuration(s)...")
            wavefunction = WaveFunction()
            if hamiltonian.x_0:
                hamiltonian.x_0 = 0.5
            return variational_monte_carlo(wavefunction, hamiltonian, num_particles, 
                                           num_samples, num_iterations,
                                           learning_rate, dof, delta,
                                           firstrun=False, target_energy=target_energy,
                                           verbose=verbose)
        
        optimizer.apply_gradients(zip(gradients, wavefunction.get_trainable_variables()))
        
        if iteration % 10 == 0 and verbose:
            print(f"Iteration {int(iteration/10+1)}: Loss = {loss_value.numpy():.3f}")
            
        if hamiltonian.x_0:
            hamiltonian.x_0 -= 0.0004

        energies.append(-loss_value.numpy())
        
    
    if hamiltonian.x_0:
        print(hamiltonian.x_0)
    
    positions = tf.concat(positions, axis=1).numpy()
    
    return -loss_value.numpy(), hamiltonian.energy, positions
