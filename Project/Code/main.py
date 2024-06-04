from runner import run_neural_network_model, run_restricted_boltzmann_model
from hamiltonians import RBM, NN

# TEMPLATE / HOW TO USE:
    #RBM:
        # run_restricted_boltzmann_model(RBM(hamiltonian, [args]), num_particles,
        # num_hidden, num_samples, num_iterations, runs, dof))
        
        # Optional arg:
            # verbose (bool)
    #NN:
        # run_neural_network_model(NN(hamiltonian, [args]), num_particles,
        # num_samples, num_iterations, runs, dof))
        
        # Optional arg:
            # verbose (bool)
        
# ACCEPTED HAMILTONIANS:
    # 'harmonic_oscillator'
        # args: omega
    # 'two_fermions'
        # args: omega
    # 'calogero_sutherland'
        # args: omega, beta
    # 'ising'
        # args: Gamma, V
    # 'heisenberg'
        # args: None

# run_restricted_boltzmann_model(RBM('harmonic_oscillator', 1), 1, 8, 250, 250, 1, 1, verbose=True)
# run_restricted_boltzmann_model(RBM('two_fermions', 1), 2, 8, 250, 250, 1, 2, verbose=True)
# run_restricted_boltzmann_model(RBM('calogero_sutherland', [1, 2]), 6, 8, 250, 250, 5, 1, verbose=True)
# run_restricted_boltzmann_model(RBM('ising', [-1, -1]), 6, 10, 250, 250, 1, 1, verbose=True)
# run_restricted_boltzmann_model(RBM('heisenberg', []), 6, 10, 250, 250, 1, 1, verbose=True)




# run_neural_network_model(NN('harmonic_oscillator', 1), 1, 500, 200, 1, 1, verbose=True, load=False)
# run_neural_network_model(NN('two_fermions', 1), 2, 200, 200, 1, 2, verbose=True, load=False)
# run_neural_network_model(NN('calogero_sutherland', [1, 2]), 6, 200, 200, 1, 1, verbose=True, load=False)
# run_neural_network_model(NN('ising', [-1, -1]), 6, 200, 200, 1, 1, verbose=True, load=False)
run_neural_network_model(NN('heisenberg', []), 6, 200, 200, 1, 1, verbose=True, load=False)
