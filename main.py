import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#TODO
#PARTICLE OBJECTS FOR SCALABILITY + TYPES ETC. obj(boson/fermion)
#wavefunction shape and size defined on call
#GPU usage
#decorator optimization

if tf.config.experimental.list_physical_devices("GPU"):
    print("Using GPU")
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)
else:
    print("Using CPU")
    

def hamiltonian(psi, x1, x2, omega=1, normalize=False):
    """
    Calculate the Hamiltonian operator for two interacting fermions (electrons).

    Args:
        psi (callable): A TensorFlow neural network estimating the wavefunction.
        x1 (tf.Tensor): Positional tensor for electron 1 with shape (N, M).
        x2 (tf.Tensor): Positional tensor for electron 2 with shape (N, M).
        omega (float): The harmonic oscillator frequency.
        normalize (bool): Whether or not to normalize psi

    Returns:
        tf.Tensor: The Hamiltonian operator applied to psi.
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

def local_energy(psi, H, x1, x2):
    """
    Calculate the local energy for the ground state.

    Args:
        psi (callable): A TensorFlow neural network estimating the wavefunction.
        H (tf.tensor): The hamiltonian operator of the wavefunction
        x1 (tf.Tensor): Positional tensor for electron 1 with shape (N, M).
        x2 (tf.Tensor): Positional tensor for electron 2 with shape (N, M).

    Returns:
        tf.Tensor: The local energy for the ground state.
    """    
    psi_val = psi(x1) * psi(x2)
    
    local_energy = tf.reduce_mean(H/ psi_val)
    
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
    def __init__(self, l2_regularization=0.05, dropout_rate=0.1):
        self.layer1 = tf.keras.layers.Dense(256, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer2 = tf.keras.layers.Dense(128, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer3 = tf.keras.layers.Dense(64, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.layer4 = tf.keras.layers.Dense(16, activation='sigmoid',
                                           kernel_regularizer=tf.keras.regularizers.l2(l2_regularization))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
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
 
    for i in range(num_samples):
         proposed_x = x[i] + delta * tf.random.normal((dof,), dtype=tf.float32)
 
         psi_current = psi(tf.expand_dims(x[i], 0))
         psi_proposed = psi(tf.expand_dims(proposed_x, 0))
         acceptance_prob = tf.minimum(1.0, tf.square(tf.abs(psi_proposed) / tf.abs(psi_current)))
 
         if tf.random.uniform(()) < acceptance_prob:
             x[i].assign(proposed_x)
 
    return x

def variational_monte_carlo(wavefunction, num_samples, num_iterations, learning_rate, dof, delta, firstrun=True, target_energy=None):
    """
    Perform Variational Monte Carlo (VMC) optimization to find the ground state energy.

    Args:
        wavefunction (WaveFunction): The wavefunction to optimize.
        num_samples (int): The number of configurations to sample.
        num_iterations (int): The number of VMC iterations.
        learning_rate (float): The initial learning rate for the optimizer.
        dof (int): The degrees of freedom in the system.
        delta (float): The Metropolis-Hastings step size.
        firstrun (bool): Flag to indicate if this is the first run (for handling NaN/inf values).
        target_energy (float): The target energy (if known).

    Returns:
        None
    """

    optimizer = tf.optimizers.Adam(learning_rate)
    x1 = tf.Variable(tf.random.normal((num_samples, dof), dtype=tf.float32), trainable=True)
    x2 = tf.Variable(tf.random.normal((num_samples, dof), dtype=tf.float32), trainable=True)


    initial_learning_rate = learning_rate
    
    energies = []
    

    for iteration in range(num_iterations):
        x1 = metropolis_hastings_update(x1, wavefunction, delta)
        x2 = metropolis_hastings_update(x2, wavefunction, delta)

        with tf.GradientTape(persistent=True) as tape:
            energy = hamiltonian(wavefunction, x1, x2)
            
            
        gradients = tape.gradient(energy, wavefunction.get_trainable_variables())

        energy = local_energy(wavefunction, energy, x1, x2)
        
        if target_energy:
            energy_difference = abs(target_energy - energy)
        else:
            energy_difference = 0
    
        
        if np.isnan(energy.numpy()).any() or np.isinf(energy.numpy()).any():
            if firstrun:
                print("NaN or inf encountered in configuration. Trying new initial configuration(s)...")
            wavefunction = WaveFunction()
            return variational_monte_carlo(wavefunction, num_samples, num_iterations, learning_rate, dof, delta, firstrun=False, target_energy=target_energy)
        
        
        updated_learning_rate = initial_learning_rate * (1.0 + 0.1 * energy_difference)


        optimizer.apply_gradients(zip(gradients, wavefunction.get_trainable_variables()))
        
        learning_rate = updated_learning_rate
        
    
        if iteration % 10 == 0:
            print(f"Iteration {int(iteration/10+1)}: Energy = {energy.numpy():.3f} a.u.")
        
        energies.append(energy.numpy())
        
    plt.plot(energies)
    plt.plot([target_energy]*len(energies))
    plt.title('Local energy of an interacting 2 fermion system')
    plt.legend(['Numerical', 'Analytical'])
    plt.show()
    print('Final energy: ', energy.numpy())

if __name__ == "__main__":
    num_samples = 200
    num_iterations = 3000
    learning_rate = 0.001
    dof = 2
    delta = 0.005
    target_energy = tf.constant(3.0, dtype=tf.float32)


    wavefunction = WaveFunction()
    
    variational_monte_carlo(wavefunction, num_samples, num_iterations, learning_rate, dof, delta, target_energy=target_energy)
    