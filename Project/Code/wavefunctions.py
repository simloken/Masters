import numpy as np

class Wavefunctions:
    def __init__(self, name):
        accepted_hamiltonians = ['harmonic_oscillator',
                                 'two_fermions', 'calogero_sutherland',
                                 'ising', 'heisenberg']
        
        if name not in accepted_hamiltonians:
            raise ValueError('Unrecognized Hamiltonian, try: ', accepted_hamiltonians)
            
    
        if name == 'harmonic_oscillator':
            self.alpha = 0.5
            self.wf = self.harmonic_oscillator
        elif name == 'two_fermions':
            self.alpha = 0.5
            self.beta = 0.5
            self.wf = self.two_fermions
        elif name == 'calogero_sutherland':
            self.alpha = 0.5
            self.beta = 1.5
            self.wf = self.calogero_sutherland
        elif name == 'ising':
            self.alpha = 0.5
            self.beta = 0.5
            self.wf = self.ising
        elif name == 'heisenberg':
            self.alpha = 1
            self.wf = self.heisenberg    
    
    def harmonic_oscillator(self, x):
        A = np.sqrt(self.alpha / np.pi)
        return A * np.exp(-0.5 * self.alpha * x**2)
    def two_fermions(self, r):
        psi = np.exp(-self.alpha * np.linalg.norm(r, axis=1)**2)
        return psi
    def calogero_sutherland(self, x):
        x = np.array(x)
        N = len(x)
        i, j = np.triu_indices(N, 1)
        product = np.prod(np.abs(x[i] - x[j])**self.beta)
        return np.exp(-0.5 * self.alpha * np.sum(x**2)) * product
    def ising(self, spins):
        spins = np.array(spins).astype(float) 
        spins = spins.reshape((spins.shape[0], spins.shape[1]))
        return np.prod(np.tanh(self.beta * spins))
    def heisenberg(self, spins):
        spins = np.array(spins)
        spins = spins.reshape(spins.shape[0], spins.shape[1])
        return np.exp(self.alpha * np.sum(spins[:-1] * spins[1:]))
    
    