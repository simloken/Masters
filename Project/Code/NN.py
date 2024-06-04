import numpy as np
import torch
import torch.nn as nn

import itertools

def loss(wavefunction, model, samples):
    """
    Calculate the loss function for Variational Monte Carlo (VMC) optimization.

    This function computes the loss used during VMC optimization to estimate the ground state energy of a quantum system.

    Args:
        wavefunction (callable): A PyTorch neural network estimating the wavefunction.
        model (object): The Hamiltonian object.
        samples (list of torch.Tensor): List of positional tensors for each particle.

    Returns:
        torch.Tensor: The energy loss value to be minimized during VMC optimization.
    """
    
    if model.name == 'calogero_sutherland': #must be sorted as they are bosons
        samples, _ = torch.sort(samples, dim=1)
    
    H_psi = model.hamiltonian(wavefunction, samples)
    
    psi_vals = wavefunction(samples)
    psi_star_vals = torch.conj(wavefunction(samples))
    psi_magnitude_squared = torch.sum(psi_vals**2) / len(samples)
    
    expectation_H = psi_star_vals * H_psi
    loss_value = torch.mean(expectation_H / psi_magnitude_squared)
        
    return loss_value

def normalize(wavefunction, samples, name):
    """
    Normalize the wavefunction using Monte Carlo estimation.

    Args:
        wavefunction (callable): A PyTorch neural network estimating the wavefunction.
        samples (list of torch.Tensor): List of positional tensors for each particle.

    Returns:
        callable: The normalized wavefunction.
    """
    if name == 'calogero_sutherland':
        samples, _ = torch.sort(samples, dim=0)
    
    psi_vals = wavefunction(samples)
    psi_magnitude_squared = psi_vals**2
    
    integral = torch.mean(psi_magnitude_squared)
    
    def normalized_wavefunction(x):
        if torch.is_complex(x):
            cintegral = integral.type(torch.complex64)
            return wavefunction(x) / torch.sqrt(cintegral)
        else:
            return wavefunction(x) / torch.sqrt(integral)
    
    return normalized_wavefunction


class NeuralNetwork(nn.Module):
    """
    A neural network model for variational quantum Monte Carlo (VMC) calculations.

    This class defines a feedforward neural network with multiple hidden layers
    for representing the wavefunction in VMC calculations.
    """
    def __init__(self, dof, regularization=0.3):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dof, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.Linear(4, 1),
            nn.Tanh()
        )
        self.regularization = regularization
        self.initialize_weights()
        
        
    def initialize_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Compute the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the neural network's prediction.
        """
        return torch.sqrt(torch.abs(self.model(x)))

class WaveFunction(nn.Module):
    """
    Wavefunction model based on a neural network.

    This class encapsulates the wavefunction model used in variational quantum
    Monte Carlo (VMC) calculations. It utilizes a neural network defined by the
    NeuralNetwork class.

    Attributes:
        NN: A neural network model (NeuralNetwork instance).
        symmetric: A boolean indicating if the wavefunction should be symmetric.
    """
    def __init__(self, particles, hamiltonian):
        super(WaveFunction, self).__init__()
        self.H_name = hamiltonian.name
        if self.H_name == 'two_fermions':
            dof = 2 * particles
            self.dof = 2
            self.particles = particles
        else:
            dof = particles
            self.dof = particles
        self.NN = NeuralNetwork(dof=dof)
        self.symmetric = hamiltonian.symmetric
        self.particles = particles

    def forward(self, x):
        """
        Compute the forward pass of the wavefunction model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor representing the wavefunction's prediction.
        """
        if self.symmetric == True:
            return self.symmetric_forward(x)
        elif self.symmetric == False:
            return self.antisymmetric_forward(x)
        else:
            return self.NN(x)

    def symmetric_forward(self, x):
        """
        Compute the symmetric forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Symmetric output.
        """
        perms = list(itertools.permutations(range(self.particles)))
        output = 0
        for perm in perms:
            permuted_x = self.permute_input(x, perm)
            output += self.NN(permuted_x)
        return output / len(perms)

    def antisymmetric_forward(self, x):
        """
        Compute the antisymmetric forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Antisymmetric output.
        """
        perms = list(itertools.permutations(range(self.particles)))
        output = 0
        for perm in perms:
            permuted_x = self.permute_input(x, perm)
            parity = self.permutation_parity(perm)
            output += parity * self.NN(permuted_x)
        
        return output / len(perms)

    def permute_input(self, x, perm):
        """
        Permute the input tensor according to the permutation.

        Args:
            x (torch.Tensor): Input tensor.
            perm (tuple): Permutation.

        Returns:
            torch.Tensor: Permuted input tensor.
        """
        x_reshaped = x.view(x.size(0), self.particles, self.dof)
        permuted_x = x_reshaped[:, perm, :]
        permuted_x = permuted_x.view(x.size(0), -1)
        return permuted_x


    @staticmethod
    def permutation_parity(perm):
        """
        Compute the parity of a permutation.

        Args:
            perm (tuple): A permutation.

        Returns:
            int: Parity of the permutation (1 for even, -1 for odd).
        """
        perm = list(perm)
        inversions = 0
        for i in range(len(perm)):
            for j in range(i + 1, len(perm)):
                if perm[i] > perm[j]:
                    inversions += 1
        return 1 if inversions % 2 == 0 else -1
    
def compute_force(x, psi):
    """
    Compute the quantum force for a given configuration.

    Args:
        x (torch.Tensor): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.

    Returns:
        torch.Tensor: The computed quantum force, shape (num_samples, dof).
    """
    x = x.requires_grad_(True)
    psi_x = psi(x)
    grad_psi = torch.autograd.grad(outputs=psi_x, inputs=x, grad_outputs=torch.ones_like(psi_x), create_graph=True)[0]
    force = 2 * grad_psi / psi_x
    return force

def compute_greens_function(x, proposed_x, F, proposed_F, delta):
    """
    Compute the Green's function for a given configuration.

    Args:
        x (torch.Tensor): The old configurations, shape (num_samples, dof).
        proposed_x (torch.Tensor): The new configurations, shape (num_samples, dof).
        F (torch.Tensor): The old quantum force, shape (num_samples, dof).
        proposed_F (torch.Tensor): The new quantum force, shape (num_samples, dof).

    Returns:
        torch.Tensor: The computed Green's function, shape (num_samples,).
    """
    G = 0.5 * (F + proposed_F) * (0.5**2 * delta * (F - proposed_F) - proposed_x + x)
    G = torch.sum(G, dim=1)
    G = torch.exp(G)
    return G

def metropolis_hastings_update(x, num_particles, psi, delta, hamiltonian):
    """
    Perform a Metropolis-Hastings update for a set of configurations.

    Args:
        x (torch.Tensor): The current configurations, shape (num_samples, dof).
        psi (callable): A function that computes the wavefunction for a given configuration.
        delta (float): The step size for the Metropolis-Hastings update.

    Returns:
        torch.Tensor: Updated configurations after Metropolis-Hastings updates, shape (num_samples, dof).
    """
    
    name = hamiltonian.name
    
    num_samples, dof_times_particles = x.shape
    dof = int(dof_times_particles / num_particles)
    
    F = compute_force(x, psi)
    
    proposed_x = x + torch.sqrt(torch.tensor(delta)) * torch.randn_like(x) + 0.5 * delta * F
    proposed_F = compute_force(proposed_x, psi)
        
    if name == 'calogero_sutherland':
        x = torch.sort(x, dim=1).values
        proposed_x = torch.sort(proposed_x, dim=1).values
      
    psi_current = psi(x)
    psi_proposed = psi(proposed_x)
    
    
    G = compute_greens_function(x, proposed_x, F, proposed_F, delta)

    if dof == 2:
        G = G.unsqueeze(-1)
        acceptance_prob = torch.min(torch.tensor(1.0), G * torch.min(torch.tensor(1.0), (psi_proposed / psi_current)**2))
    else:
        acceptance_prob = torch.min(torch.tensor(1.0), G * torch.squeeze(torch.min(torch.tensor(1.0), (psi_proposed / psi_current)**2), dim=-1)).unsqueeze(1)
     
    random_numbers = torch.rand(num_samples)
    accept_mask = random_numbers.unsqueeze(1) < acceptance_prob
    x = torch.where(accept_mask, proposed_x, x)
    return x

def metropolis_hastings_spin_update(samples, psi, hamiltonian):
    """
    Perform a Metropolis-Hastings update for a set of spin configurations.

    Args:
        samples (torch.Tensor): The current spin configurations, shape (num_samples, num_particles, 1).
        psi (callable): A function that computes the wavefunction for a given configuration.

    Returns:
        torch.Tensor: Updated spin configurations after Metropolis-Hastings updates, shape (num_samples, num_particles, 1).
    """
    M, N = samples.shape

    rand_indices = torch.randint(0, N, (M,))

    updates = -samples[range(M), rand_indices]

    proposed_samples = samples.clone()
    proposed_samples[range(M), rand_indices] = updates

    psi_current = psi(samples)
    psi_proposed = psi(proposed_samples)

    acceptance_prob = torch.min(torch.tensor(1.0), (psi_proposed / psi_current) ** 2)

    random_numbers = torch.rand(M, 1)
    accept_mask = random_numbers < acceptance_prob
    samples = torch.where(accept_mask, proposed_samples, samples)

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
    
    optimizer = torch.optim.Adam(wavefunction.NN.parameters(), lr=learning_rate)
    # torch.autograd.set_detect_anomaly(True)                     
    
    if hamiltonian.spin: #lattice particle spin
        samples = [torch.where(torch.rand(num_samples, dof) > 0.5, 0.5, -0.5)
         for _ in range(num_particles)]

    else: #free particles
        samples = [torch.randn(num_samples, dof)
                 for _ in range(num_particles)]
        
    samples = torch.cat(samples, dim=-1)
    
        
    energies = []
        
    for iteration in range(num_iterations):
        
        samples = samples.detach()
        
        psi = normalize(wavefunction, samples, hamiltonian.name)

        if not hamiltonian.spin: #free particles
            samples = metropolis_hastings_update(samples, num_particles, psi, delta, hamiltonian)
                
        if hamiltonian.spin: #lattice spin
            psi = normalize(wavefunction, samples, hamiltonian.name)
            samples = metropolis_hastings_spin_update(samples, psi, hamiltonian)
            
        optimizer.zero_grad()
        psi = normalize(wavefunction, samples, hamiltonian.name) #renormalize after moves (recalculating normalization constant)
        loss_value = loss(psi, hamiltonian, samples)

        loss_value.backward()
        optimizer.step()
        
        if iteration % 10 == 0 and verbose:
            print(f'{iteration}: {loss_value.item()}')
        elif iteration+1 == num_iterations and verbose:
            print(f'{iteration+1}: {loss_value.item()}')
            
        energies.append(loss_value.item())
        if debug:
            for name, param in wavefunction.NN.named_parameters():
                if param.grad is not None:
                    print(f'Gradient for {name}: {param.grad.norm().item()}')
        
        if hamiltonian.x_0: # ugly
            factor = (hamiltonian.x_0_initial-hamiltonian.x_0_minimum)/num_iterations
            if hamiltonian.x_0 > hamiltonian.x_0_minimum:  
                hamiltonian.x_0 -= factor
                if hamiltonian.x_0 < hamiltonian.x_0_minimum:
                    hamiltonian.x_0 = hamiltonian.x_0_minimum
            else:
                hamiltonian.x_0 = hamiltonian.x_0_minimum
                
    if debug:
        import matplotlib.pyplot as plt
        with torch.no_grad():
            x_values = torch.linspace(-5, 5, 1000).view(-1, dof)
            psi_values = torch.square(psi(x_values))
            plt.figure(figsize=(10, 6))
            plt.plot(x_values.numpy(), psi_values.numpy(), label='Wavefunction')
            plt.xlabel('x')
            plt.ylabel(r'\psi(x)')
            plt.title(f'Wavefunction ψ(x) for {hamiltonian.name}')
            plt.legend()
            plt.show()
            
    return loss_value.item(), hamiltonian, np.real(energies), samples
