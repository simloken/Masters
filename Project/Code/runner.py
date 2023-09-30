import numpy as np

import time

from RBM import RBM
from NN import variational_monte_carlo as VMCNN
from RBM import variational_monte_carlo as VMCRBM
from NN import WaveFunction

def run_neural_network_model(hamiltonian, num_particles, num_samples, num_iterations,
                             runs, dof, target_energy=None, verbose=False):
    """
    Run a neural network-based model to estimate the ground state energy of a quantum system.

    Args:
        hamiltonian (function): The hamiltonian of the model
        num_particles (int): The number of particles in the quantum system.
        num_samples (int): The number of Monte Carlo samples to generate.
        num_iterations (int): The number of training iterations for the neural network.
        runs (int): The number of runs to perform to estimate the mean energy.
        dof (int): Degrees of freedom.
        target_energy (float, optional): The target energy for the system (if known). Default is None.
        verbose (bool, optional): If True, print detailed information during the runs. Default is False.

    Returns:
        None

    """
    
    delta = 0.005
    learning_rate = 0.002

    energy_storage = []
    t0 = time.time()
    for k in range(runs):
        
        trun = time.time()
        wavefunction = WaveFunction()
        
        energy_storage.append(VMCNN(wavefunction, hamiltonian, num_samples, num_iterations,
                                                      learning_rate, dof, delta, target_energy=target_energy, verbose=verbose))
        if verbose:
            print(f"Run #{k+1} time: {(time.time() - trun):.2f}s")
        
    print(f"Mean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")
    
    
    
def run_restricted_boltzmann_model(hamiltonian, num_particles, num_samples, num_iterations,
                                   num_hidden, runs, dof, verbose=False):
    """
    Run a Restricted Boltzmann Machine (RBM) model to estimate the ground state energy of a quantum system.

    Args:
        hamiltonian (function): The hamiltonian of the model
        num_particles (int): The number of particles in the quantum system.
        num_samples (int): The number of Monte Carlo samples to generate.
        num_iterations (int): The number of training iterations for the RBM.
        num_hidden (int): The number of hidden units in the RBM.
        runs (int): The number of runs to perform to estimate the mean energy.
        dof (int): Degrees of freedom.
        verbose (bool, optional): If True, print detailed information during the runs. Default is False.

    Returns:
        None

    """
    t0 = time.time()
    omega = 1
    energy_storage = []

    num_visible = num_particles * dof
    for k in range(runs):
        rbm = RBM(num_visible, num_hidden)
        
        VMCRBM(hamiltonian, rbm, num_visible, num_samples, num_iterations, omega, dof) 
    
        ground_state_energy = []
        for _ in range(num_samples):
            sample = np.random.randn(num_visible)
            local_energy = hamiltonian(sample, omega, dof)
            ground_state_energy.append(local_energy)
    
        estimated_energy = np.mean(ground_state_energy)
        if verbose:
            print(f"Energy: {estimated_energy:.3f} a.u.")
        energy_storage.append(estimated_energy)
    print(f"Mean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")