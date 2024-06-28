import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x)
def f(x):
    return np.exp(-x**2)

# Normalization constant for f(x)
def normalize(f, x):
    
    fx = f(x)
    f_squared = np.square(abs(fx))
    
    integral = np.sum(f_squared)
    
    print(np.sum((fx/np.sqrt(integral))**2))
        
    return np.sqrt(integral)

# Metropolis-Hastings algorithm
def metropolis_hastings(f, num_samples, delta):
    x = np.random.normal(size=(num_samples,))
    acceptance_count = 0

    for i in range(1, num_samples):
        x_prop = x[i-1] + np.random.normal(scale=delta)
        acceptance_ratio = f(x_prop) / f(x[i-1])
        
        if np.random.rand() < acceptance_ratio:
            x[i] = x_prop
            acceptance_count += 1
        else:
            x[i] = x[i-1]
    
    acceptance_rate = acceptance_count / num_samples
    return x, acceptance_rate

# Parameters
num_samples = 10000
delta = 1.0


# Generate samples using Metropolis-Hastings
samples, acceptance_rate = metropolis_hastings(f, num_samples, delta)

# Calculate normalization constant
norm_const = normalize(f, samples)

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot histogram of the samples for unnormalized function
ax1.hist(samples, bins=50, density=True, alpha=0.6, color='g')
x = np.linspace(-3, 3, 1000)
ax1.plot(x, f(x), 'r-', lw=2)
ax1.set_title('Unnormalized Function $f(x)$')
ax1.set_xlabel('x')
ax1.set_ylabel('Probability Density')

# Plot histogram of the samples for normalized function
ax2.hist(samples, bins=50, density=True, alpha=0.6, color='g')
ax2.plot(x, f(x)/norm_const, 'b-', lw=2)
ax2.set_title('Normalized Function f(x)')
ax2.set_xlabel('x')
ax2.set_ylabel('Probability Density')

# Show the plots
plt.tight_layout()
plt.show()

# Print acceptance rate
print(f"Acceptance rate: {acceptance_rate}")
