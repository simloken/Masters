import numpy as np
import jax

import datetime
import winsound
import os.path


from NN import WaveFunction
from RBM import RBM as RBMWaveFunction
from hamiltonians import RBM, NN
from NN import variational_monte_carlo as VMCNN
from RBM import variational_monte_carlo as VMCRBM
from analysis import relative_error
from ansatz import pre_train_NN


def findbestNN(hamiltonian, num_particles, num_samples, num_iterations, runs,
               delta_values, learning_rate_values, sound=True):
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
        sound (bool): Whether or not to play a sound when the computation is finished

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
                    wavefunction = WaveFunction(num_particles, h)
                    # if not os.path.isfile(f'./weights/{h.name}_weights.h5'):
                    #     pre_train_NN(wavefunction, h.name, num_particles, dof)
                    #     wavefunction = WaveFunction(num_particles, h.name) #redefine after pre-training
                    
                    # if os.path.isfile(f'./weights/{h.name}_weights.h5'):
                    #     wavefunction.NN.model.load_weights(f'./weights/{h.name}_weights.h5')
                    
                    energy, hamiltonian, energies, positions = VMCNN(wavefunction, h, num_particles, num_samples, num_iterations,
                                                                  learning_rate, dof, delta)
                    energy_list.append(energy)
                    true_energy = hamiltonian.energy
                    if isinstance(true_energy, str):
                        true_energy_str = true_energy
                        true_energy = float(true_energy[2:])

                    
                    if h.x_0:
                        h.x_0 = 0.5
                    
                err = relative_error(np.mean(energy_list), true_energy)
                if abs(err) < abs(best_err):
                    best_err = err
                    best_params = (delta, learning_rate)
                    best_energy = np.mean(energy_list)
                    best_energy_std = np.std(energy_list)
                    
        current_directory = os.getcwd()

        parent_directory = os.path.dirname(current_directory)
        
        data_directory = os.path.join(parent_directory, 'Data')
        
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        
        file_path = os.path.join(data_directory, 'findbestNN.txt')

        with open(file_path, 'a') as f:
            f.write(f"Model: {H} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"Energy: Mean={best_energy}, Std={best_energy_std}\n")
            f.write(f"Relative Error:{best_err}\n\n")
            
        if sound == True:
            duration = 1000
            freq = 440
            winsound.Beep(freq, duration)
            
            
def findbestRBM(hamiltonian, num_particles, num_hidden, num_samples, num_iterations, runs,
                delta_values, learning_rate_values, sound=True):
    """
    Find the best combination of delta and learning_rate for a given Hamiltonian.

    Args:
        hamiltonian (string or list): Hamiltonian(s) to iterate over
        num_particles (int): The number of particles in the quantum system.
        num_hidden(int): The number of hidden units in the RBM
        num_samples (int): The number of Monte Carlo samples to generate.
        num_iterations (int): The number of training iterations for the neural network.
        runs (int): The number of runs to perform to estimate the mean energy.
        dof (int): Degrees of freedom.
        delta_values (list): List of delta values to iterate over.
        learning_rate_values (list): List of learning_rate values to iterate over.
        sound (bool): Whether or not to play a sound when the computation is finished

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
        best_err = float('inf')
        best_params = None
        i, j = 0, 0
        num_visible = num_particles * dof
        for delta in delta_values:
            i += 1
            j = 0
            for learning_rate in learning_rate_values:
                j += 1
                print(f"Delta {i}, Eta {j}")
                energy_list = []
                for k in range(runs):
                    h = RBM(H, 'default')
                    key = jax.random.PRNGKey(0)
                    rbm = RBMWaveFunction(num_visible, num_hidden, key, dof, learning_rate, False, H)
                    energy, hamiltonian, energies, positions = VMCRBM(rbm, h, num_particles, num_samples, num_iterations,
                                                    learning_rate, dof, delta, verbose=False, debug=False)
                                    
                    energy_list.append(energy)

                    true_energy = hamiltonian.energy
                    if h.x_0:
                        h.x_0 = 0.5
                        
                    if isinstance(true_energy, str):
                        true_energy_str = true_energy
                        true_energy = float(true_energy[2:])
                    
                err = relative_error(np.mean(energy_list), true_energy)
                if abs(err) < abs(best_err):
                    best_err = err
                    best_params = (delta, learning_rate)
                    best_energy = np.mean(energy_list)
                    best_energy_std = np.std(energy_list)
                    
        current_directory = os.getcwd()

        parent_directory = os.path.dirname(current_directory)
        
        data_directory = os.path.join(parent_directory, 'Data')
        
        if not os.path.exists(data_directory):
            os.makedirs(data_directory)
        
        file_path = os.path.join(data_directory, 'findbestRBM.txt')
                    
        

        with open(file_path, 'a') as f:
            f.write(f"Model: {H} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Best params: {best_params}\n")
            f.write(f"Energy: Mean={best_energy}, Std={best_energy_std}\n")
            f.write(f"Relative Error:{best_err}\n\n")
            
        if sound == True:
            duration = 1000
            freq = 440
            winsound.Beep(freq, duration)

            
N = 1
delta_min, delta_max = 0.001, 0.1
learning_rate_min, learning_rate_max = 0.0001, 0.001
deltas = np.linspace(delta_min, delta_max, N)
learning_rates = np.linspace(learning_rate_min, learning_rate_max, N)

findbestNN('harmonic_oscillator', 1, 100, 200, 1, deltas, learning_rates)
# findbestNN('two_fermions', 2, 400, 200, 3, deltas, learning_rates)
# findbestNN('ising', 6, 500, 500, 3, [1], learning_rates)
# findbestNN('heisenberg', 6, 250, 1500, 3, [1], learning_rates)

# findbestRBM('harmonic_oscillator', 1, 3, 200, 200, 1, deltas, learning_rates)
# findbestRBM('two_fermions', 2, 8, 10, 200, 1, deltas, learning_rates)

