import numpy as np

class Energies:
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
    
    def ising(N):
        """
        Return a very rough approximate for the true energy of a 1D Ising Model for
        Gamma = 1, V = -1
        
        Args:
            N (int): The number of particles
            
        Returns:
            str: Approximation of the true energy
        """
        approx = -N/2 * np.sqrt(6.5)
        return '~ %g' %(approx)