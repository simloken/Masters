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
        
# OPTIONAL ARGS:
    # verbose
        # whether or not to print various information underway
    # load
        # whether or not to pre-load/pre-train a model
    # debug
        # whether or not to print debug information such as gradients or generate animations for sample distribution
        
        
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
        
    # args can also be 'default'
        
    

# run_restricted_boltzmann_model(RBM('harmonic_oscillator', 1), 1, 4, 100, 500, 10, 1, verbose=True, load=False)
# run_restricted_boltzmann_model(RBM('two_fermions', 1), 2, 4, 100, 250, 10, 2, verbose=True, load=False)
# run_restricted_boltzmann_model(RBM('calogero_sutherland', [1, 2]), 3, 4, 100, 200, 1, 1, verbose=True, load=False)
# run_restricted_boltzmann_model(RBM('ising', [-1, -1]), 6, 4, 250, 250, 10, 1, verbose=True, load=False)
# run_restricted_boltzmann_model(RBM('heisenberg', []), 6, 4, 250, 250, 10, 1, verbose=True, load=False)




# run_neural_network_model(NN('harmonic_oscillator', 1), 1, 1000, 1000, 1, 1, verbose=True, load=False)
# run_neural_network_model(NN('two_fermions', 1), 2, 1000, 1000, 1, 2, verbose=True, load=False)
run_neural_network_model(NN('calogero_sutherland', [1, 2]), 20, 1000, 1000, 3, 1, verbose=True, load=False)
# run_neural_network_model(NN('ising', [-1, -1]), 6, 1000, 1000, 1, 1, verbose=True, load=False)
# run_neural_network_model(NN('heisenberg', []), 6, 1000, 1000, 10, 1, verbose=True, load=False)



