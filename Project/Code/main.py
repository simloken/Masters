from runner import run_neural_network_model, run_restricted_boltzmann_model
from hamiltonians import RBM, NN

# run_restricted_boltzmann_model(RBM.calogero_sutherland, 6, 1000, 1000, 25, 1, 1)
# run_restricted_boltzmann_model(RBM.two_fermions, 2, 100, 100, 25, 10, 2)
# run_neural_network_model(NN.two_fermions, 2, 400, 1500, 2, 2, target_energy=3)
run_neural_network_model(NN.calogero_sutherland, 6, 100, 100, 1, 1)