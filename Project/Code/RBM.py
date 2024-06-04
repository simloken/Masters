import numpy as np
import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax.example_libraries import optimizers


def loss(params, wavefunction, model, samples, dof, skip=False):
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
    
    wavefunction = normalize(wavefunction, samples, dof, model.name, params)
    
    H_psi = model.hamiltonian(wavefunction, samples)
    
    
    psi_vals = wavefunction(samples, dof)
    psi_magnitude_squared = jnp.sum(jnp.square(psi_vals))
    psi_star_vals = jnp.conj(wavefunction(samples, dof))
        
    expectation_H = jnp.multiply(psi_star_vals, H_psi)
    


    loss_value = jnp.mean(expectation_H / psi_magnitude_squared)
    
    return loss_value

def normalize(wavefunction, samples, dof, name, params=None):
    """
    Normalize the wavefunction using Monte Carlo estimation.

    Args:
        wavefunction (callable): A JAX neural network estimating the wavefunction.
        samples (list of jnp.ndarray): List of positional tensors for each particle.

    Returns:
        callable: The normalized wavefunction.
    """
    if name == 'calogero_sutherland':
        samples = jnp.sort(samples, axis=0)
            
    psi_vals = wavefunction(samples, dof)
    psi_magnitude_squared = jnp.square(psi_vals)
    integral = jnp.mean(psi_magnitude_squared) * jnp.size(samples)
    
    
    def normalized_wavefunction(x, dof, grad=False, params=params):
        return wavefunction(x, dof, grad, params) / jnp.sqrt(integral)
 
    return normalized_wavefunction



class RBM:
    def __init__(self, num_visible, num_hidden, key, learning_rate):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.key = key
        self.params = self.initialize_params()
        self.opt_init, self.opt_update, self.get_params = optimizers.adam(learning_rate)
        self.opt_state = self.opt_init(self.params)
        self.norm_const = 1
        
    def initialize_params(self):
        key_W, key_a, key_b = jax.random.split(self.key, 3)
        W = 0.1*jax.random.normal(key_W, (self.num_visible, self.num_hidden))
        # a = jax.random.normal(key_a, (self.num_visible,))
        # b = jax.random.normal(key_b, (self.num_hidden,))
        a = jnp.zeros((self.num_visible,))
        b = jnp.zeros((self.num_hidden,))
        return W, a, b
    
    def forward(self, v):
        W, a, b = self.params
        return sigmoid(jnp.dot(v, W) + b)
    
    def backward(self, h):
        W, a, b = self.params
        return sigmoid(jnp.dot(h, W.T) + a)
    
    def wavefunction(self, x, dof, grad=False, params=None):
        if params is not None:
            self.params = params
        W, a, b = self.params
        sigma = 1.0
        scaling_factor = 4 * sigma**2 * self.num_visible
    
        def compute_wavefunction(x_single):
            exponent = -jnp.sum((x_single - a)**2 / scaling_factor)
            prod_term = 1 + jnp.exp(b + jnp.dot(x_single, W) / sigma**2)
            log_prod_term = jnp.sum(jnp.log(prod_term))
            return jnp.abs(jnp.exp(exponent + 0.5 * log_prod_term))
        
        if grad:
            # Single input
            return compute_wavefunction(x)
        # Batched input
        exponent = -jnp.sum((x - a)**2 / scaling_factor, axis=1)
        prod_term = 1 + jnp.exp(b + jnp.dot(x, W) / sigma**2)
        log_prod_term = jnp.sum(jnp.log(prod_term), axis=1)
        return jnp.abs(jnp.exp(exponent + 0.5 * log_prod_term))  
        
    
    def update(self, grads):
        self.opt_state = self.opt_update(0, grads, self.opt_state)
        self.params = self.get_params(self.opt_state)
    
    def compute_gradients(self, wavefunction, hamiltonian, samples, dof):
        loss_value, grads = jax.value_and_grad(loss)(self.params, wavefunction, hamiltonian, samples, dof)
        return loss_value, grads



def compute_force(x, psi, dof):
    """
    Compute the quantum force for a given configuration.

    Args:
        x (jnp.ndarray): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.

    Returns:
        jnp.ndarray: The computed quantum force, shape (num_samples, dof).
    """
    psi_x = psi(x, dof)
    grad_psi = jax.vmap(jax.grad(lambda x_i: psi(x_i, dof, grad=True).squeeze()))(x)
    force = 2 * grad_psi / psi_x[:, None]
    return force

def compute_greens_function(x, proposed_x, F, proposed_F, delta):
    """
    Compute the Green's function for a given configuration.

    Args:
        x (jnp.ndarray): The old configurations, shape (num_samples, dof).
        proposed_x (jnp.ndarray): The new configurations, shape (num_samples, dof).
        F (jnp.ndarray): The old quantum force, shape (num_samples, dof).
        proposed_F (jnp.ndarray): The new quantum force, shape (num_samples, dof).

    Returns:
        jnp.ndarray: The computed Green's function, shape (num_samples,).
    """
    G = 0.5 * (F + proposed_F) * (0.5 * delta * (F - proposed_F) - proposed_x + x)
    G = jnp.sum(G, axis=1)
    G = jnp.exp(G)
    return G


def metropolis_hastings_update(x, num_particles, psi, delta, hamiltonian):
    """
    Perform a Metropolis-Hastings update for a set of configurations.

    Args:
        x (np.ndarray): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.
        delta (float): The step size for the Metropolis-Hastings update.

    Returns:
        np.ndarray: Updated configurations after Metropolis-Hastings updates, shape (num_samples, dof).
    """
    
    name = hamiltonian.name
    
    num_samples, dof_times_particles = np.shape(x)
    dof = int(dof_times_particles / num_particles)
    

    F = compute_force(x, psi, dof)
    proposed_x = x + np.sqrt(delta) * np.random.normal(size=x.shape) + 0.5 * delta * F
    proposed_F = compute_force(proposed_x, psi, dof)
        
    if name == 'calogero_sutherland':
        x = np.sort(x, axis=1)
        proposed_x = np.sort(proposed_x, axis=1)
      
    psi_current = psi(x, dof)
    psi_proposed = psi(proposed_x, dof)
    
    
    G = compute_greens_function(x, proposed_x, F, proposed_F, delta)
    
    if dof == 2:
        acceptance_prob = np.minimum(1.0, G * np.minimum(1.0, np.square(psi_proposed / psi_current)))
        acceptance_prob = jnp.expand_dims(acceptance_prob, axis=-1)
    else:
        acceptance_prob = np.minimum(1.0, np.reshape(G * np.minimum(1.0, np.square(psi_proposed / psi_current)), (-1,1)))
        
    # print('Mean acceptance probability:', jnp.mean(acceptance_prob))
        
    random_numbers = np.random.uniform(size=(num_samples,))
    accept_mask = random_numbers[:, np.newaxis] < acceptance_prob
    
    x = np.where(accept_mask, proposed_x, x)
    return x

def metropolis_hastings_spin_update(samples, psi, key):
    """
    Perform a Metropolis-Hastings update for a set of spin configurations.

    Args:
        samples (jnp.ndarray): The current spin configurations, shape (num_samples, num_particles, 1).
        psi (callable): A function that computes the wavefunction for a given configuration.
        key (jax.random.PRNGKey): A JAX PRNG key for random number generation.

    Returns:
        jnp.ndarray: Updated spin configurations after Metropolis-Hastings updates, shape (num_samples, num_particles, 1).
    """
    
    M, N = samples.shape
    
    key, subkey = jax.random.split(key)
    rand_indices = jax.random.randint(subkey, shape=(M,), minval=0, maxval=N)
    
    proposed_samples = samples.at[jnp.arange(M), rand_indices].set(-samples[jnp.arange(M), rand_indices])

    psi_current = psi(samples, 1)
    psi_proposed = psi(proposed_samples, 1)
    
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
        wavefunction (RBM): The RBM wavefunction to optimize.
        hamiltonian (function): The Hamiltonian of the system.
        num_particles (int): The number of particles in the system.
        num_samples (int): The number of configurations to sample.
        num_iterations (int): The number of VMC iterations.
        learning_rate (float): The initial learning rate for the optimizer.
        dof (int): The degrees of freedom in the system.
        delta (float): The Metropolis-Hastings step size.
        verbose (bool): Whether or not to print progress.

    Returns:
        energy (float): The final ground state energy of the system.
    """
    
    keys = jax.random.split(rbm.key, num_particles)
    
    if hamiltonian.spin:  # lattice particle spin
        samples = [jnp.where(jax.random.uniform(k, (num_samples, dof)) > 0.5, 0.5, -0.5) for k in keys]
    else:  # free particles
        samples = [jax.random.normal(k, (num_samples, dof)) for k in keys]
    
    samples = jnp.concatenate(samples, axis=-1)
        
    energies = []
        
    for iteration in range(num_iterations):
        psi = normalize(rbm.wavefunction, samples, dof, hamiltonian.name)

        if not hamiltonian.spin:  # free particles
            samples = metropolis_hastings_update(samples, num_particles, psi, delta, hamiltonian)
                
        if hamiltonian.spin:  # lattice spin
            samples = metropolis_hastings_spin_update(samples, psi, rbm.key)
            
        
        psi = normalize(rbm.wavefunction, samples, dof, hamiltonian.name) #renormalize after moves (recalculating normalization constant)
        
        loss_value, grads = rbm.compute_gradients(rbm.wavefunction, hamiltonian, samples, dof)
        
        if iteration % 10 == 0 and verbose:
            print(f'{iteration}: {loss_value}')
        elif iteration + 1 == num_iterations and verbose:
            print(f'{iteration+1}: {loss_value}')
        
        rbm.update(grads)
        energies.append(loss_value)
        
        if hamiltonian.x_0:  # ugly
            factor = (hamiltonian.x_0_initial - hamiltonian.x_0_minimum) / num_iterations
            if hamiltonian.x_0 > hamiltonian.x_0_minimum:  
                hamiltonian.x_0 -= factor
                if hamiltonian.x_0 < hamiltonian.x_0_minimum:
                    hamiltonian.x_0 = hamiltonian.x_0_minimum
            else:
                hamiltonian.x_0 = hamiltonian.x_0_minimum


    return loss_value, hamiltonian, np.real(energies), samples


    
