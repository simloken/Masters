import numpy as np
import matplotlib.pyplot as plt


hbar, m, omega = 1, 1, 1


def psi_true(x):
    return (m*omega/(np.pi*hbar))**(1/4)*np.exp(-m*omega*x**2/(2*hbar))

def phi_1(x):
    a = np.sqrt(m*omega/(np.sqrt(2)*hbar))
    A = np.sqrt(a)
    return A * np.exp(-a*abs(x))

def phi_2(x): 
    a = m*omega/(np.sqrt(7)*hbar)
    A = np.sqrt(16*np.sqrt(a)/(5*np.pi))
    return A/(1 + a*x**2)**2


x = np.linspace(-10, 10, 1000)

plt.plot(x, psi_true(x), label='True Wave Function', color='k')
plt.plot(x, phi_1(x), label = r'$\phi_1$ - Exponential', color = 'c')
plt.plot(x ,phi_2(x), label = r'$\phi_2$ - Lorentzian', color = 'm')
plt.legend()
plt.xlabel('x')
plt.ylabel(r'$\Psi(x)$')
plt.title('Different wave functions as a function of x')
plt.show()