import numpy as np
import tensorflow as tf
import gc

import datetime
import winsound
import os.path


from NN import WaveFunction
from RBM import RBM
from hamiltonians import RBM, NN
from NN import variational_monte_carlo as VMCNN
from RBM import variational_monte_carlo as VMCRBM
from analysis import relative_error
from ansatz import pre_train_NN


def findbestNN(hamiltonian, num_particles, num_samples, num_iterations, runs, delta_values, learning_rate_values, sound=True):
    """
    Find the best combination of delta and learning_rate for a given Hamiltonian.

    Args:
        hamiltonian (string or list): Hamiltonian(s) to iterate over
        num_particles (int): The number of particles in the quantum system.
        num_samples (int): The number of Monte Carlo samples to generate.
        num_iterations (int): The number of training iterations for the neural network.
        runs (int): The number of runs to perform to estimate the mean energy.
        dof (int): Degrees of freedom.
        delta_values (list): List of delta values to iterate over.
        learning_rate_values (list): List of learning_rate values to iterate over.

    Returns:
        None
    """
    if hamiltonian == 'all':
        hamiltonian= ['harmonic_oscillator', 'two_fermions', 'calogero_sutherland', 'ising', 'heisenberg']
    else:
        hamiltonian = [hamiltonian]

    for H in hamiltonian:
        if H == 'two_fermions':
            dof = 2
            num_particles = 2
        else:
            dof = 1
        tf.keras.backend.clear_session()
        gc.collect()
        best_err = float('inf')
        best_params = None
        i, j = 0, 0
        for delta in delta_values:
            i += 1
            j = 0
            for learning_rate in learning_rate_values:
                j += 1
                print(f"Delta {i}, Eta {j}")
                energy_list = []
                for k in range(runs):
                    h = NN(H, 'default')
                    wavefunction = WaveFunction(num_particles, h.name)
                    # if not os.path.isfile(f'./weights/{h.name}_weights.h5'):
                    #     pre_train_NN(wavefunction, h.name, num_particles, dof)
                    #     wavefunction = WaveFunction(num_particles, h.name) #redefine after pre-training
                    
                    # if os.path.isfile(f'./weights/{h.name}_weights.h5'):
                    #     wavefunction.NN.model.load_weights(f'./weights/{h.name}_weights.h5')
                    
                    energy, hamiltonian, energies, positions = VMCNN(wavefunction, h, num_particles, num_samples, num_iterations,
                                                                  learning_rate, dof, delta)
                    energy_list.append(energy)

                    
                    if h.x_0:
                        h.x_0 = 0.5
                    
                    tf.keras.backend.clear_session()
                    gc.collect()
                err = relative_error(np.mean(energy_list), hamiltonian.energy)
                if abs(err) < abs(best_err):
                    best_err = err
                    best_params = (delta, learning_rate)
                    best_energy = np.mean(energy_list)
                    best_energy_std = np.std(energy_list)

        with open('findbestNN.txt', 'a') as f:
            f.write(f"Model: {H} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"Energy: Mean={best_energy}, Std={best_energy_std}\n")
            f.write(f"Relative Error:{best_err}\n\n")
            
        if sound == True:
            duration = 1000
            freq = 440
            winsound.Beep(freq, duration)
            
            
            
def tunehypersNN(hamiltonian, num_particles, num_samples, num_iterations, runs, dof, totune, tuneboth=False):
    ...
            
N = 5           
delta_min, delta_max = 0.001, 0.05
learning_rate_min, learning_rate_max = 0.0001, 0.001
deltas = np.linspace(delta_min, delta_max, N)
learning_rates = np.linspace(learning_rate_min, learning_rate_max, N)

findbestNN('harmonic_oscillator', 1, 200, 200, 5, deltas, learning_rates)
