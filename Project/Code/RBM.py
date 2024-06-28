import numpy as np
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
import itertools

from tools import load_weights


def loss(W, a, b, wavefunction, model, samples, dof):
    """
    Calculate the loss function for Variational Monte Carlo (VMC) optimization.

    Args:
        wavefunction (callable): A function estimating the wavefunction.
        model (object): The Hamiltonian object.
        samples (list of jnp.ndarray): List of positional tensors for each particle.

    Returns:
        jnp.ndarray: The energy loss value to be minimized during VMC optimization.
    """
    
    if model.name == 'calogero_sutherland':  # must be sorted as they are bosons
        samples = jnp.sort(samples, axis=1)
    
    wavefunction = normalize(wavefunction, samples, dof, model.name, W, a, b)
    
    H_psi = model.hamiltonian(wavefunction, samples, W, a, b)
    psi_vals = wavefunction(samples, dof, 0, W, a, b)
        
    local_energy = H_psi / psi_vals
    local_energy = jnp.mean(local_energy)
        
    return local_energy

def normalize(wavefunction, samples, dof, name, W, a, b):
    """
    Normalize the wavefunction using Monte Carlo estimation.

    Args:
        wavefunction (callable): A JAX neural network estimating the wavefunction.
        samples (list of jnp.ndarray): List of positional tensors for each particle.

    Returns:
        callable: The normalized wavefunction.
    """
    if name == 'calogero_sutherland':
        samples = jnp.sort(samples, axis=1)
            
    psi_vals = wavefunction(samples, dof, 0, W, a, b)
    psi_magnitude_squared = jnp.square(psi_vals)
    integral = jnp.mean(psi_magnitude_squared) #god knows why this must be mean and not sum but otherwise it grows/lowers exponentially
    
    
    def normalized_wavefunction(x, dof, grad, W, a, b):
        if grad == 0:
            return wavefunction(x, dof, grad, W, a, b) / jnp.sqrt(integral)
        else: #ensures gradients are not also divided by jnp.sqrt(integral)
            return wavefunction(x, dof, grad, W, a, b)
 
    return normalized_wavefunction



class RBM:
    def __init__(self, num_particles, num_hidden, key, dof, learning_rate, pre_trained, name):
        """
        Initialize the Restricted Boltzmann Machine (RBM) model.

        Args:
            num_particles (int): Number of particles.
            num_hidden (int): Number of hidden units.
            key (jax.random.PRNGKey): PRNG key for random number generation.
            dof (int): Degrees of freedom.
            learning_rate (float): Learning rate for the optimizer.
            pre_trained (bool): Whether to load pre-trained weights.
            name (str): Name for loading pre-trained weights.
        """
        self.num_hidden = num_hidden
        self.particles = num_particles
        self.dof = dof
        self.num_visible = num_particles*dof
        self.key = key
        if not pre_trained:
            self.params = self.initialize_params()
        else:
            self.params = load_weights(f'{name}')
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.params)

        
    def initialize_params(self):
        """
        Initialize the parameters of the RBM.

        Returns:
            tuple: Initialized weights and biases (W, a, b).
        """
        key_W, key_a, key_b = jax.random.split(self.key, 3)
        W = 0.01 * jax.random.normal(key_W, (self.num_visible, self.num_hidden))
        # a = jax.random.normal(key_a, (self.num_visible,))
        # b = jax.random.normal(key_b, (self.num_hidden,))
        a = jnp.zeros(self.num_visible)
        b = jnp.zeros(self.num_hidden)
        return W, a, b
    
    def wavefunction(self, X, dof, grad, W, a, b):
        """
        Compute the wavefunction and its gradients.

        Args:
            X (jax.numpy.ndarray): Input configurations.
            dof (int): Degrees of freedom.
            grad (int): Gradient order (0 for no gradient, 1 for first-order, 2 for second-order).
            W (jax.numpy.ndarray): Weight matrix.
            a (jax.numpy.ndarray): Visible biases.
            b (jax.numpy.ndarray): Hidden biases.

        Returns:
            jax.numpy.ndarray: Computed wavefunction or its gradient.
        """
        sigma2 = 1
        psiFactor1 = jnp.sum((X - a) ** 2, axis=1)
        psiFactor1 = jnp.exp(-psiFactor1 / (2.0 * sigma2))
            
        Q = b + (X @ W)/sigma2
    
        psiFactor2 = jnp.prod(1 + jnp.exp(Q), axis=1)
        
        
    
        psi = psiFactor1 * psiFactor2
        if grad == 1: #1st order
            dPsi_dX = jax.jacobian(self.wavefunction)(X, dof, 0, W, a, b).sum(axis=1)
            return dPsi_dX
        elif grad == 2: # 2nd order derivative
            def u(x):
                return jnp.sum(self.wavefunction(x, dof, 0, W, a, b))
            
            d2Psi_dX2 = jax.jacobian(jax.grad(u))(X).sum(axis=1).sum(axis=1).sum(axis=1)
            
                                    
            return d2Psi_dX2
                
        return jnp.sqrt(psi/self.Z)
    
    def symmetric_forward(self, x, dof, grad, W, a, b):
        """
        Compute the symmetric forward pass.

        Args:
            x (jax.numpy.ndarray): Input configurations.
            dof (int): Degrees of freedom.
            grad (int): Gradient order.
            W (jax.numpy.ndarray): Weight matrix.
            a (jax.numpy.ndarray): Visible biases.
            b (jax.numpy.ndarray): Hidden biases.

        Returns:
            jax.numpy.ndarray: Symmetrized wavefunction.
        """
        perms = list(itertools.permutations(range(self.particles)))
        perms_tensor = jnp.array(perms, dtype=jnp.int32)
        
        output = 0
        for perm in perms_tensor:
            permuted_x = self.permute_input(x, perm)
            output += self.wavefunction(permuted_x, self.dof, 0, W, a, b)
        
        return output / len(perms)

    def antisymmetric_forward(self, x, dof, grad, W, a, b):
        """
        Compute the antisymmetric forward pass.

        Args:
            x (jax.numpy.ndarray): Input configurations.
            dof (int): Degrees of freedom.
            grad (int): Gradient order.
            W (jax.numpy.ndarray): Weight matrix.
            a (jax.numpy.ndarray): Visible biases.
            b (jax.numpy.ndarray): Hidden biases.

        Returns:
            jax.numpy.ndarray: Antisymmetrized wavefunction.
        """
        perms = list(itertools.permutations(range(self.particles)))
        perms_tensor = jnp.array(perms, dtype=jnp.int32)
        parity = self.permutation_parity(perms_tensor)
        
        output = 0
        for perm, p in zip(perms_tensor, parity):
            permuted_x = self.permute_input(x, perm)
            output += p * self.wavefunction(permuted_x, self.dof, 0, W, a, b)
        
        return output / len(perms)

    def permute_input(self, x, perm):
        """
        Permute the input configurations.

        Args:
            x (jax.numpy.ndarray): Input configurations.
            perm (jax.numpy.ndarray): Permutation indices.

        Returns:
            jax.numpy.ndarray: Permuted input configurations.
        """
        if self.dof > 1:
            x_reshaped = x.reshape(x.shape[0], self.particles, self.dof)
            permuted_x = x_reshaped[:, perm, :]
            permuted_x = permuted_x.reshape(x.shape[0], -1)
        else:
            x_reshaped = x.reshape(x.shape[0], self.particles)
            permuted_x = x_reshaped[:, perm]
            permuted_x = permuted_x.reshape(x.shape[0], -1)
        return permuted_x

    @staticmethod
    def permutation_parity(perms):
        """
        Compute the parity of permutations.

        Args:
            perms (jax.numpy.ndarray): Permutation indices.

        Returns:
            jax.numpy.ndarray: Parities of the permutations.
        """
        inversions = jnp.zeros(perms.shape[0], dtype=jnp.int32)
        for i in range(perms.shape[1]):
            for j in range(i + 1, perms.shape[1]):
                inversions += (perms[:, i] > perms[:, j]).astype(jnp.int32)
        parity = 1 - 2 * (inversions % 2)
        return parity

    
    def partition_function(self, W, a, b):
        """
        Compute the partition function.

        Args:
            W (jax.numpy.ndarray): Weight matrix.
            a (jax.numpy.ndarray): Visible biases.
            b (jax.numpy.ndarray): Hidden biases.

        Returns:
            None
        """
        sigma = 1.0
        scaling_factor = 2 * sigma**2
        
        def energy(x, h):
            term1 = jnp.sum((x - a)**2 / scaling_factor)
            term2 = jnp.sum(b * h)
            term3 = jnp.sum((x @ W) * h / sigma**2)
            return term1 - term2 - term3
    
        x_values = jnp.linspace(-10, 10, 100).reshape(-1, 1)
        if self.num_visible != 1:
            x_values = jnp.tile(x_values, (1, self.num_visible))

        h_values = jnp.array([[0, 1]] * self.num_hidden).T
    
        Z = 0
        for h in h_values:
            for x in x_values:
                Z += jnp.exp(-energy(x, h))
    
        self.Z = Z
        
    def psi(self, X, dof, grad, W, a, b):
        """
        Compute the wavefunction based on symmetry.

        Args:
            X (jax.numpy.ndarray): Input configurations.
            dof (int): Degrees of freedom.
            grad (int): Gradient order.
            W (jax.numpy.ndarray): Weight matrix.
            a (jax.numpy.ndarray): Visible biases.
            b (jax.numpy.ndarray): Hidden biases.

        Returns:
            jax.numpy.ndarray: Computed wavefunction.
        """
        if self.symmetric == True:
            return self.symmetric_forward(X, dof, grad, W, a, b)
        elif self.symmetric == False:
            return self.antisymmetric_forward(X, dof, grad, W, a, b)
        else:
            return self.wavefunction(X, dof, grad, W, a, b)
        
    
    def update(self, step, grads):
        """
        Update the parameters using the optimizer.

        Args:
            step (int): Current optimization step.
            grads (tuple): Gradients of the parameters.

        Returns:
            None
        """
        self.opt_state = self.opt_update(step, grads, self.opt_state)
        self.params = self.get_params(self.opt_state)
    
    def compute_gradients(self, wavefunction, hamiltonian, samples, dof):
        """
        Compute the gradients of the loss function.

        Args:
            wavefunction (callable): Wavefunction function.
            hamiltonian (callable): Hamiltonian function.
            samples (jax.numpy.ndarray): Input samples.
            dof (int): Degrees of freedom.

        Returns:
            tuple: Loss value and gradients.
        """
        W, a, b = self.params
        grad_loss = jax.value_and_grad(loss, argnums=(0, 1, 2))
        loss_value, grads = grad_loss(W, a, b, wavefunction, hamiltonian, samples, dof)
        if self.debug:
            print(grads)
        grads = jax.tree_map(lambda g: jnp.clip(g, -1, 1), grads)
        return loss_value, grads




def compute_force(x, psi, dof, rbm):
    """
    Compute the quantum force for a given configuration.

    Args:
        x (jnp.ndarray): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.
        dof (int): Degrees of freedom.
        rbm (RBM): The RBM model instance.

    Returns:
        jnp.ndarray: The computed quantum force, shape (num_samples, dof).
    """
    W, a, b = rbm.params
    psi_x = psi(x, dof, 0, W, a, b)
    grad_psi = psi(x, dof, 1, W, a, b)

    force = 2 * grad_psi / psi_x[:, None]
    return force

def compute_greens_function(x, proposed_x, F, delta, dof, name):
    """
    Compute the Green's function for a given configuration.

    Args:
        x (jnp.ndarray): The old configurations, shape (num_samples, dof).
        proposed_x (jnp.ndarray): The new configurations, shape (num_samples, dof).
        F (jnp.ndarray): The old quantum force, shape (num_samples, dof).
        delta (float): The time step.
        dof (int): Degrees of freedom.
        name (str): Name of the system or model.

    Returns:
        jnp.ndarray: The computed Green's function, shape (num_samples,).
    """
    num_samples, dof = x.shape
    N = dof
    D = 0.5
    
    normalization_factor = 1 / (4 * jnp.pi * D * delta)**(0.5 * N)
    
    if name == 'calogero_sutherland':
        diff = proposed_x.sum(axis=1) - x.sum(axis=1) - D * delta * F
    else:
        diff = proposed_x - x- D * delta * F
    exponent = -jnp.sum(diff**2) / (4 * D * delta)
    
    G = normalization_factor * jnp.exp(exponent)
    return G


def metropolis_hastings_update(x, num_particles, psi, delta, hamiltonian, rbm):
    """
    Perform a Metropolis-Hastings update for a set of configurations.

    Args:
        x (jnp.ndarray): The current configurations, shape (num_samples, dof).
        num_particles (int): Number of particles in the system.
        psi (callable): A function that computes the wavefunction for a given configuration.
        delta (float): The step size for the Metropolis-Hastings update.
        hamiltonian (object): The Hamiltonian of the system.
        rbm (RBM): The RBM model instance.

    Returns:
        jnp.ndarray: Updated configurations after Metropolis-Hastings updates, shape (num_samples, dof).
    """
    W, a, b = rbm.params
    name = hamiltonian.name
    
    num_samples, dof_times_particles = x.shape
    dof = int(dof_times_particles / num_particles)
    
    F = compute_force(x, psi, dof, rbm)
    
    indx = []
    
    reroll_gauss = np.arange(0, dof_times_particles, dof)
    
    for i in range(num_particles):
        indx.append(np.random.randint(0, num_samples, (1,)))
    proposed_x = x.copy()
    for idx in indx: #update random index
        for i in range(dof_times_particles): #of particle
            if i in reroll_gauss:
                guass = np.random.normal() #ensures same random number along all dof
                proposed_x = proposed_x.at[idx,i].set(x[idx,i] + jnp.sqrt(delta) * guass + 0.5 * delta * F[idx, i])
    
    if name == 'calogero_sutherland':
        x = jnp.sort(x, axis=1)
        proposed_x = jnp.sort(proposed_x, axis=1)
      
    psi_current = psi(x, dof, 0, W, a, b)
    psi_proposed = psi(proposed_x, dof, 0, W, a, b)
    
    psi_c_squared = jnp.square(psi_current).squeeze()
    psi_p_squared = jnp.square(psi_proposed).squeeze()
    
    G = compute_greens_function(x, proposed_x, F, delta, dof, name)
    

    g_psi_c_squared = G*psi_c_squared
    g_psi_p_squared = G*psi_p_squared
    
    ratio = g_psi_p_squared/g_psi_c_squared
        
    random_numbers = np.random.normal()
    accept_mask = random_numbers < ratio

    updated_x = jnp.where(accept_mask[:, jnp.newaxis], proposed_x, x)
    
    return updated_x

def metropolis_hastings_spin_update(samples, psi, rbm):
    """
    Perform a Metropolis-Hastings update for a set of spin configurations.

    Args:
        samples (jnp.ndarray): The current spin configurations, shape (num_samples, num_particles, 1).
        psi (callable): A function that computes the wavefunction for a given configuration.
        rbm (RBM): The RBM model instance.

    Returns:
        jnp.ndarray: Updated spin configurations after Metropolis-Hastings updates, shape (num_samples, num_particles, 1).
    """
    W, a, b = rbm.params
    M, N = samples.shape
    key = rbm.key
    key, subkey = jax.random.split(key)
    rand_indices = jax.random.randint(subkey, shape=(M,), minval=0, maxval=N)
    
    proposed_samples = samples.at[jnp.arange(M), rand_indices].set(-samples[jnp.arange(M), rand_indices])

    psi_current = psi(samples, 1, 0, W, a, b)
    psi_proposed = psi(proposed_samples, 1, 0, W, a, b)
    
    acceptance_prob = jnp.minimum(1.0, (psi_proposed / psi_current) ** 2)
    
    key, subkey = jax.random.split(key)
    random_numbers = jax.random.uniform(subkey, (M,))
    accept_mask = random_numbers < acceptance_prob

    samples = jnp.where(accept_mask[:, None], proposed_samples, samples)
    
    return samples

def variational_monte_carlo(rbm, hamiltonian, num_particles, num_samples,
                            num_iterations, learning_rate, dof, delta,
                            verbose=None, debug=False):
    """
    Perform Variational Monte Carlo (VMC) optimization to find the ground state energy.

    Args:
        rbm (RBM): The RBM wavefunction to optimize.
        hamiltonian (object): The Hamiltonian of the system.
        num_particles (int): The number of particles in the system.
        num_samples (int): The number of configurations to sample.
        num_iterations (int): The number of VMC iterations.
        learning_rate (float): The initial learning rate for the optimizer.
        dof (int): The degrees of freedom in the system.
        delta (float): The Metropolis-Hastings step size.
        verbose (bool): Whether or not to print progress.
        debug (bool): Whether or not to print debug information.

    Returns:
        tuple: The final ground state energy, Hamiltonian, energy history, and final samples.
    """
    
    keys = jax.random.split(rbm.key, num_particles)
    
    rbm.symmetric = hamiltonian.symmetric
    jax.config.update("jax_enable_x64", True) #else grads become NaN
    rbm.debug = debug
    
    if hamiltonian.spin:  # lattice particle spin
        samples = [jnp.where(jax.random.uniform(k, (num_samples, dof)) > 0.5, 0.5, -0.5) for k in keys]
    else:  # free particles
        samples = [jax.random.normal(k, (num_samples, dof)) for k in keys]
    
    samples = jnp.concatenate(samples, axis=-1)
        
    energies = []
            
    for iteration in range(num_iterations):
        W, a, b = rbm.params
        rbm.partition_function(W, a, b)
        psi = normalize(rbm.psi, samples, dof, hamiltonian.name, W, a, b)

        if not hamiltonian.spin:  # free particles
            samples = metropolis_hastings_update(samples, num_particles, psi, delta, hamiltonian, rbm)
                
        if hamiltonian.spin:  # lattice spin
            samples = metropolis_hastings_spin_update(samples, psi, rbm)
            
        psi = normalize(rbm.psi, samples, dof, hamiltonian.name, W, a, b) #renormalize after moves (recalculating normalization constant)
        
        loss_value, grads = rbm.compute_gradients(psi, hamiltonian, samples, dof)
                        
        if iteration % 10 == 0 and verbose:
            print(f'{iteration}: {loss_value}')
        elif iteration + 1 == num_iterations and verbose:
            print(f'{iteration+1}: {loss_value}')
        
        rbm.update(iteration, grads)
        energies.append(loss_value)
        
        if hamiltonian.x_0: # ugly
            if iteration > num_iterations/4:
                factor = (hamiltonian.x_0_initial-hamiltonian.x_0_minimum)/(num_iterations/4)
                if hamiltonian.x_0 > hamiltonian.x_0_minimum:  
                    hamiltonian.x_0 -= factor
                    if hamiltonian.x_0 < hamiltonian.x_0_minimum:
                        hamiltonian.x_0 = hamiltonian.x_0_minimum
                else:
                    hamiltonian.x_0 = hamiltonian.x_0_minimum
                



    return loss_value, hamiltonian, np.real(energies), samples


    
