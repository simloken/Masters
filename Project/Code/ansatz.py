import torch
import torch.optim as optim
import numpy as np
import jax
import jax.numpy as jnp

from wavefunctions import Wavefunctions

import os

def trial_wavefunction(name):
    """
    Imports trial wave function ansatz

    Args:
        name (str or lst of str): The model to save the pre-trained network to

    Returns:
        wavefunction (callable): The wave function ansatz

    """
    init = Wavefunctions(name)
    return init.wf

def loss_fn(y_true, y_pred):
    """
    Calculates the mean squared loss

    Args:
        y_true (list-like): The values for the ansatz
        y_pred (list-like): The values for the prediction
    Returns:
        loss (float): The mean squared error

    """
    return torch.mean((y_true - y_pred) ** 2)

def pre_train_NN(model, name, num_particles, dof):
    """
    Pre-train a Neural Network (NN) model according to an ansatz.

    Args:
        model (object): The Neural Network model to generate weights for
        name (str or lst of str): The model to save the pre-trained network to
        num_particles (int): The number of particles in the quantum system.
        dof (int): Degrees of freedom.

    Returns:
        None

    """
    
    lattice = ['ising', 'heisenberg']
        
    if name in lattice:
        x = np.where(np.random.uniform(size=(1000, num_particles)) > 0.5, 0.5, -0.5)
        x = torch.tensor(x, dtype=torch.float32)
    elif name == 'two_fermions':
        x = [torch.randn(100, dof) for _ in range(num_particles)]
        x = torch.cat(x, dim=-1)
    
    optimizer = optim.Adam(model.NN.parameters())

    y_values = trial_wavefunction(name)(x)

    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_values, y_pred)
        loss.backward()
        optimizer.step()

    torch.save(model.NN.state_dict(), f'./weights/{name}_weights.pth')



def pre_train_RBM(name, num_particles, num_hidden, dof, key):
    """
    Pre-train a Restricted Boltzmann Machine (RBM) model according to an ansatz.

    Args:
        name (str or lst of str): The model to save the pre-trained network to
        num_particles (int): The number of particles in the quantum system.
        num_hidden (int): The number of hidden units in the RBM.
        dof (int): Degrees of freedom.
        key (jax.random.key()): The key to use for (pseudo) random number generation

    Returns:
        None

    """
    lattice = ['ising', 'heisenberg']
    keys = jax.random.split(key, num_particles)
    if name in lattice:
        samples = jnp.array([jnp.where(jax.random.uniform(k, (1000, dof)) > 0.5, 0.5, -0.5) for k in keys])
    else:
        samples = jnp.array([jax.random.normal(k, (1000, dof)) for k in keys])
        
    y_values = trial_wavefunction(name)(samples)
    
    key_W, key_a, key_b = jax.random.split(keys[0], 3)
    W = 0.01 * jax.random.normal(key_W, (num_particles*dof, num_hidden))
    a = jnp.zeros(num_particles*dof)
    b = jnp.zeros(num_hidden)
        
    def partition_function(W, a, b):
        sigma = 1.0
        scaling_factor = 2 * sigma**2
        
        def energy(x, h):
            term1 = jnp.sum((x - a)**2 / scaling_factor)
            term2 = jnp.sum(b * h)
            term3 = jnp.sum((x @ W) * h / sigma**2)
            return term1 - term2 - term3
    
        x_values = jnp.linspace(-10, 10, 100).reshape(-1, 1)
        if num_particles*dof != 1:
            x_values = jnp.tile(x_values, (1, num_particles*dof))

        h_values = jnp.array([[0, 1]] * num_hidden).T
    
        Z = 0
        for h in h_values:
            for x in x_values:
                Z += jnp.exp(-energy(x, h))
    
        return Z
    
    def y(X, dof, W, a, b):
        sigma2 = 1
        psiFactor1 = jnp.sum((X - a) ** 2, axis=1)
        psiFactor1 = jnp.exp(-psiFactor1 / (2.0 * sigma2))
            
        Q = b + (X @ W)/sigma2
    
        psiFactor2 = jnp.prod(1 + jnp.exp(Q), axis=1)
        
        psi = psiFactor1 * psiFactor2
        Z = partition_function(W, a, b)
        return jnp.sqrt(psi/Z)
    
    @jax.jit
    def loss_fn(y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)
    
    @jax.jit
    def update(params, X, y_true, lr=0.01):
        W, a, b = params
        y_pred = y(X, dof, W, a, b)
        loss = loss_fn(y_true, y_pred)
        grads = jax.grad(lambda p: loss_fn(y_true, y(X, dof, *p)))(params)
        new_params = [param - lr * grad for param, grad in zip(params, grads)]
        return new_params, loss
    
    params = [W, a, b]
    for i in range(1000):  # number of iterations
        for sample, y_true in zip(samples, y_values):
            params, loss = update(params, sample, y_true)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
    
    # Save the weights
    W, a, b = params
    if not os.path.exists('./weights'):
        os.makedirs('./weights')
    np.savez(f'./weights/{name}_weights.npz', W=W, a=a, b=b)