#from runner import run_neural_network_model, run_restricted_boltzmann_model
from runner import run_restricted_boltzmann_model
from hamiltonians import RBM #, NN

# TEMPLATE / HOW TO USE:
    #RBM:
        # run_restricted_boltzmann_model(RBM(hamiltonian, hyperparameters), num_particles,
        # num_samples, num_iterations, num_hidden, runs, dof))
    #NN:
        # run_neural_network_model(NN(hamiltonian, hyperparameters), num_particles,
        # num_samples, num_iterations, runs, dof))
        
# ACCEPTED HAMILTONIANS:
    # 'two_fermions'
        # args: omega
    # 'calogero_sutherland'
        # args: omega, beta
    # 'ising'
        # args: Gamma, V
    # 'heisenberg'
        # args: None

# run_restricted_boltzmann_model(RBM('two_fermions', 1), 2, 1000, 1000, 3, 1, 2)
# run_restricted_boltzmann_model(RBM('calogero_sutherland', [1, 2]), 6, 1000, 1000, 33, 1, 1)
run_restricted_boltzmann_model(RBM('ising', [-1, -1]), 6, 1000, 1000, 6, 1, 1)

# run_neural_network_model(NN('two_fermions', 1), 2, 5000, 2000, 1, 2)
# run_neural_network_model(NN('calogero_sutherland', [1, 2]), 6, 5000, 1000, 1, 1)
#run_neural_network_model(NN('ising', [-1, -1]), 6, 5000, 1000, 1, 1)
# run_neural_network_model(NN('heisenberg', []), 6, 5000, 1000, 1, 1)
