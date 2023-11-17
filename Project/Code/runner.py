import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gc

import time

from NN import WaveFunction
from RBM import RBM
from NN import variational_monte_carlo as VMCNN
from RBM import variational_monte_carlo as VMCRBM
from analysis import plot_particle_density

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
    for k in range(runs):
        
        trun = time.time()
        wavefunction = WaveFunction()
        
        energy, true_energy, positions = VMCNN(wavefunction, hamiltonian, num_particles, num_samples, num_iterations,
                                                      learning_rate, dof, delta,
                                                      target_energy=target_energy, verbose=verbose)
        energy_storage.append(energy)
        
        plot_particle_density(positions, dof)
        plt.show()
        
        if hamiltonian.x_0:
            hamiltonian.x_0 = 0.5
        
        tf.keras.backend.clear_session()
        gc.collect()
        
        
        
        if verbose:
            print(f"Run #{k+1} time: {(time.time() - trun):.2f}s")

    
    print((energy - true_energy)/true_energy)   
                
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
        
        samples = VMCRBM(model, rbm, num_visible, num_samples, num_iterations, dof)
    
        ground_state_energy = []
        
        for sample in samples:
            local_energy = model.hamiltonian(sample, dof)
            ground_state_energy.append(local_energy)
    
        estimated_energy = np.mean(ground_state_energy)
        if verbose:
            print(f"Energy: {estimated_energy:.3f} a.u.")
        
        
        energy_storage.append(estimated_energy)
        
        plot_particle_density(samples, dof)
        plt.show()
        
        if model.x_0:
            model.x_0 = 0.5
        
    
    print(f"Mean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"True energy of system: {rbm.energy} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")