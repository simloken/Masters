import numpy as np
import torch
from jax import random
import matplotlib.pyplot as plt

import time
import datetime
import os.path

from NN import WaveFunction
from RBM import RBM
from NN import variational_monte_carlo as VMCNN
from RBM import variational_monte_carlo as VMCRBM
from analysis import plot_particle_density, relative_error
from ansatz import pre_train_NN

def run_neural_network_model(hamiltonian, num_particles, num_samples, num_iterations,
                             runs, dof, delta=0.01, learning_rate=0.005,
                             load=True, verbose=False, debug=False):
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
    
    if torch.cuda.is_available():
        print("Using GPU")
    else:
        print("Using CPU")
        


    energy_storage = []
    energy_evolution = []
    t0 = time.time()
    print('Starting:', datetime.datetime.now().strftime('%H:%M:%S'))
    
    
    for k in range(runs):
        
        trun = time.time()
        wavefunction = WaveFunction(num_particles, hamiltonian)
        weights_path = f'./weights/{hamiltonian.name}_weights.pth'
    
        
        if load:
            if not os.path.isfile(weights_path):
                if verbose:
                    print(f'No pre-trained model found. Pre-training new model for {hamiltonian.name}')
                pre_train_NN(wavefunction, hamiltonian.name, num_particles, dof)
                wavefunction = WaveFunction(num_particles, hamiltonian)  # Redefine after pre-training
        
            if os.path.isfile(weights_path):
                try:
                    wavefunction.NN.load_state_dict(torch.load(weights_path))
                    if verbose:
                        print(f'Loaded pre-trained model made at {time.ctime(os.path.getmtime(weights_path))}')
                except Exception as e:
                    if verbose:
                        print(f'Incompatible pre-trained model detected, pre-training new model! Error: {e}')
                    pre_train_NN(wavefunction, hamiltonian.name, num_particles, dof)
                    wavefunction = WaveFunction(num_particles, hamiltonian)  # Redefine after pre-training
                    wavefunction.NN.load_state_dict(torch.load(weights_path))
                    
                    
        energy, hamiltonian, energies, positions = VMCNN(wavefunction, hamiltonian, num_particles, num_samples, num_iterations,
                                                      learning_rate, dof, delta,
                                                      verbose=verbose, debug=debug)
        
        true_energy = hamiltonian.energy
        energy_storage.append(energy)
        
        energy_evolution.append(energies)
                
        if hamiltonian.has_plots:
            plot_particle_density(positions, dof)
            plt.show()
        
        if hamiltonian.x_0:
            hamiltonian.x_0 = 0.5
            
        
        
        if runs > 1 and verbose == True:
            if k == 0:
                print('====================')
            print(f"Run #{k+1}\nEnergy: {energy:.3f} a.u.\nTime: {(time.time() - trun):.2f}s")
    
    if runs > 1 and verbose == True:
        print('====================')

    
                    
    
    
    for i in range(len(energy_storage)):
        plt.plot(energy_evolution[i][:], label=f"Run {i+1}")
    if isinstance(true_energy, str):
        true_energy_str = true_energy
        true_energy = float(true_energy[2:])
    plt.plot([true_energy]*len(energy_evolution[0][:]), label='True Energy', color='k')
    plt.ylim([true_energy-5, true_energy+5])
    plt.xlim([0, num_iterations])
    plt.grid(True)
    plt.title(f'Energy evolution over {len(energy_storage)} runs for {hamiltonian.name}')
    plt.legend()
    plt.show()
    
    print(f"\nMean energy over {runs} runs: {np.mean(energy_storage):.3} ± {np.std(energy_storage):.3} a.u.")
    print(f"Mean relative error over {runs} runs: {relative_error(np.mean(energy_storage), true_energy)}")
    if 'true_energy_str' in locals():
        print(f"True energy of system: {true_energy_str} a.u.")
    else:
        print(f"True energy of system: {true_energy} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s\n")
    
def run_restricted_boltzmann_model(hamiltonian, num_particles, num_hidden, 
                                   num_samples, num_iterations,
                             runs, dof, delta=0.005, learning_rate=0.01,
                             load=True, verbose=False, debug=False):
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
    energy_storage = []
    num_visible = num_particles * dof
    ground_state_energy = []
    t0 = time.time()
    print('Starting:', datetime.datetime.now().strftime('%H:%M:%S'))
    for k in range(runs):
        
        trun = time.time()
        key = random.PRNGKey(0)
        rbm = RBM(num_visible, num_hidden, key, learning_rate)
        energy, hamiltonian, energies, positions = VMCRBM(rbm, hamiltonian, num_particles, num_samples, num_iterations,
                                        learning_rate, dof, delta, verbose=verbose, debug=debug)
    
        ground_state_energy.append(energy)
        
    
        estimated_energy = np.mean(ground_state_energy)
        if runs > 1:
            print(f"Run #{k+1}\nEnergy: {estimated_energy:.3f} a.u.\nTime: {(time.time() - trun):.2f}s")
        
        
        energy_storage.append(estimated_energy)
        if hamiltonian.has_plots:
            plot_particle_density(positions, dof)
            plt.show()
        
    true_energy = hamiltonian.energy
    print(f"\nMean energy over {runs} runs: {np.mean(energy_storage)} a.u.")
    print(f"Mean relative error over {runs} runs: {relative_error(np.mean(energy_storage), true_energy)}")
    print(f"True energy of system: {hamiltonian.energy} a.u.")
    print(f"Total run time: {(time.time() - t0):.2f}s\nAverage run time: {((time.time() - t0)/runs):.2f}s")