### Code Folder Structure

This folder contains all of the code

#### Subfolders:

- **Deprecated**: Contains outdated code that was used during earlier stages of the project.
- **Examples**: Contains example scripts used as illustrations in the thesis.
- **netket**: Includes a method for approximating a cubic fit of a 1D Heisenberg model using NetKet
- **weights**: Can store weights of models if needed.

#### Python Files:

- **analysis.py**: Tools for analyzing experimental results.
- **ansatz.py**: Methods for pre-training neural network models.
- **energies.py**: Contains analytical solutions to the systems under study.
- **findbest.py**: Performs grid search to optimize system configurations.
- **hamiltonians.py**: Defines Hamiltonians for neural network (NN) and restricted Boltzmann machine (RBM) implementations.
- **main.py**: Main script orchestrating the overall execution.
- **NN.py**: Computation and methods for the NN approach.
- **RBM.py**: Computation and methods for the RBM approach.
- **runner.py**: Serves as a bridge between `main.py` and `RBM/NN.py`.
- **tools.py**: Contains utility functions and tools.
- **wavefunctions.py**: Defines wave functions used in the ansatz.
