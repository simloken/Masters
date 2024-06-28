import numpy as np
import os

class Energies:
    
    def harmonic_oscillator(omega):
        """
        Return the true energy of a harmonic oscillator in one dimension
        
        Returns:
            float: The true energy
            
        """
        
        return 1/2 * omega
    
    def two_fermions():
        """
        Return the true energy of a two fermion system in two dimensions
        
        Returns:
            float: 3
        """
        return 3.0

    def calogero_sutherland(N, omega, beta):
        """
        Return the true energy calogero sutherland system
        
        Args:
            N (int): The number of particles in the system
            omega (float): The harmonic oscillator frequency
            beta (float): Interaction parameter
        
        Returns:
            float: The true energy
        """
        return 0.5*N*omega*(1 + beta*(N-1))
    
    def ising(N, J, h):
        """
        Compute the ground state energy of a 1D Transverse Field Ising Model with N particles using Pfeuty's method.

        Args::
            N (int): Number of particles.
            J (float): Coupling constant.
            h (float): Transverse magnetic field.

        Returns:
            float: Ground state energy.
        """
        # Define the momentum values
        k = np.array([(2 * np.pi * (n - 0.5)) / N for n in range(1, N + 1)])
        
        # Compute the dispersion relation
        epsilon_k = np.sqrt((2 * J * np.cos(k))**2 + (2 * h)**2)
        
        # Ground state energy is the sum of -1/2 * epsilon_k for all k
        ground_state_energy = -0.5 * np.sum(epsilon_k)
        
        return -ground_state_energy
    
    def heisenberg(L):
        """
        Return a very rough approximate for the true energy of a 1D
        Antiferromagnetic Heisenberg model using a cubic approximation
        from NetKet ground states for different L:{2, 24}
        
        Args:
            L (int): The number of spins
            
        Returns:
            str: Approximation of the true energy
        """
        
        file_path = os.path.join("..", "Data", "heisenberg_approx.txt")
    
        # Read the parameters from the file
        params = {}
        with open(file_path, "r") as f:
            for line in f:
                key, value = line.strip().split(": ")
                params[key] = float(value)
        
        # Extract parameters
        a = params.get("a")
        b = params.get("b")
        c = params.get("c")
        d = params.get("d")
        
        approx = a*L**3+(b)*L**2+(c)*L+d
        return '~ %g' %(-approx)