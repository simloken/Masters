import numpy as np
import matplotlib.pyplot as plt

def psi(x):
    return np.exp(-(x - 0.5)**2 / 0.05) + 0.5 * np.exp(-(x - 1.5)**2 / 0.1) # arbitrary wave function

def normalize(psi):
    norm = np.sqrt(np.trapz(np.abs(psi)**2, dx=0.01))
    return psi / norm

A = 2/3

x = np.linspace(0, 2, 1000)

psi_normalized = normalize(psi(x))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(x, np.abs(psi_normalized)**2, label='Before Measurement', color='k')
plt.axvline(x=A, color='r', linestyle='--', label='Measurement at A')
plt.title('Wavefunction Before Measurement')
plt.ylabel(r'$|\Psi(x)|^2$', rotation=0)
plt.xticks([A], ['A'])
plt.legend()

psi_collapsed = np.zeros_like(psi_normalized)
psi_collapsed[np.argmin(np.abs(x - A))] = 1.0

plt.subplot(1, 2, 2)
plt.plot(x, np.abs(psi_collapsed)**2, label='After Measurement', color='k')
plt.title('Wavefunction After Measurement')
plt.ylabel(r'$|\Psi_A(x)|^2$', rotation=0)
plt.xticks([A], ['A'])
plt.legend()

plt.tight_layout()

plt.show()
