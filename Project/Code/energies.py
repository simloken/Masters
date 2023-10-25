
class Energies:
    def two_fermions():
        """
        Return the true energy of a two fermion system in two dimensions
        
        Returns:
            int: 3
        """
        return 3

    def calogero_sutherland(N, omega, beta):
        """
        Return the true energy calogero_sutherland system
        
        Args:
            N (int): The number of particles in the system
            omega (float): The harmonic oscillator frequency
            beta (float): Interaction parameter
        
        Returns:
            float: The true energy
        """
        return 0.5*N*omega*(1 + beta*(N-1))