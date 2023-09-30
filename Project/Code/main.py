from runner import run_neural_network_model, run_restricted_boltzmann_model
from hamiltonians import RBM, NN

run_restricted_boltzmann_model(RBM.two_fermions, 2, 1000, 5000, 3, 5, 2)
run_neural_network_model(NN.two_fermions, 2, 400, 1500, 20, 2, target_energy=3)