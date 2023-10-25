import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gc

import time

from NN import WaveFunction
from RBM import RBM
from NN import variational_monte_carlo as VMCNN
from RBM import variational_monte_carlo as VMCRBM

def run_neural_network_model(hamiltonian, num_particles, num_samples, num_iterations,
                             runs, dof, target_energy=None, verbose=False):
    """
    Run a neural network-based model to estimate the ground state energy of a quantum system.

    Args:
        model (object): The model of the hamiltonian
        num_particles (int): The number of particles in the quantum system.
        num_samples (int): The number of Monte Carlo samples to generate.
        num_iterations (int): The number of training iterations for the neural network.
        runs (int): The number of runs to perform to estimate the mean energy.
        dof (int): Degrees of freedom.
        params (list or float): Hyperparameters to pass to the Hamiltonian
        target_energy (float, optional): The target energy for the system (if known). Default is None.
        verbose (bool, optional): If True, print detailed information during the runs. Default is False.

    Returns:
        None

    """
    
    if tf.config.experimental.list_physical_devices("GPU"):
        print("Using GPU")
        tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices("GPU")[0], True)
    else:
        print("Using CPU")
        
    tf.keras.backend.clear_session()
    gc.collect()
    
    delta = 0.001
    learning_rate = 0.05

    energy_storage = []
    t0 = time.time()
    plt.figure()
    for k in range(runs):
        
        trun = time.time()
        wavefunction = WaveFunction()
        
        energy, energies, true_energy = VMCNN(wavefunction, hamiltonian, num_particles, num_samples, num_iterations,
                                                      learning_rate, dof, delta,
                                                      target_energy=target_energy, verbose=verbose)
        energy_storage.append(energy)
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        plt.plot(energies, label=f'run: {k+1}')
        
        
        if verbose:
            print(f"Run #{k+1} time: {(time.time() - trun):.2f}s")
    
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Energy [a.u.]')
    plt.title(f'Energy evolution over {runs} runs')
    plt.show()
                    
    print(f"Mean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"True energy of system: {true_energy} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")
    
    
    
def run_restricted_boltzmann_model(model, num_particles, num_samples, num_iterations,
                                   num_hidden, runs, dof, verbose=False):
    """
    Run a Restricted Boltzmann Machine (RBM) model to estimate the ground state energy of a quantum system.

    Args:
        model (object): The model of the hamiltonian
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
    energy_storage = []
    num_visible = num_particles * dof
    
    for k in range(runs):
        rbm = RBM(num_visible, num_hidden)
        
        VMCRBM(model, rbm, num_visible, num_samples, num_iterations, dof) 
    
        ground_state_energy = []
        for _ in range(num_samples):
            sample = np.random.randn(num_visible)
            local_energy = model.hamiltonian(sample, dof)
            ground_state_energy.append(local_energy)
    
        estimated_energy = np.mean(ground_state_energy)
        if verbose:
            print(f"Energy: {estimated_energy:.3f} a.u.")
        
        
        energy_storage.append(estimated_energy)
        
    
    print(f"Mean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"True energy of system: {rbm.energy} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")