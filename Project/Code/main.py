from runner import run_neural_network_model, run_restricted_boltzmann_model
from hamiltonians import RBM, NN

run_restricted_boltzmann_model(RBM('two_fermions', 1), 2, 1000, 500, 25, 2, 2)
# run_restricted_boltzmann_model(RBM('calogero_sutherland', [1, 2]), 6, 1000, 1000, 25, 1, 1)
# run_neural_network_model(NN('two_fermions', 1), 2, 5000, 1000, 1, 2)
# run_neural_network_model(NN('calogero_sutherland', [1, 2]), 6, 5000, 1000, 1, 1)