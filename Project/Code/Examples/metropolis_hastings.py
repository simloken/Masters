import numpy as np
import matplotlib.pyplot as plt

def target_distribution(x):
    """Target probability distribution function: Normal distribution N(0, 1)"""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def proposal_distribution(x, step_size=0.5):
    """Proposal distribution: Uniform distribution U(x - step_size, x + step_size)"""
    return np.random.uniform(x - step_size, x + step_size)

def metropolis_hastings(target_dist, proposal_dist, initial_sample, iterations, num_samples, step_size=0.5):
    history = []
    current_samples = initial_sample

    for _ in range(iterations):
        new_samples = []
        for current_sample in current_samples:
            proposed_sample = proposal_dist(current_sample, step_size)
            acceptance_ratio = target_dist(proposed_sample) / target_dist(current_sample)

            if acceptance_ratio >= 1 or np.random.rand() < acceptance_ratio:
                current_sample = proposed_sample

            new_samples.append(current_sample)

        current_samples = np.array(new_samples)
        history.append(current_samples.reshape(num_samples, 1))

    return np.array(history)

iterations = 300
num_samples = 1000
initial_sample = np.random.uniform(-10, 10, (num_samples,))
step_size = 0.5

samples_history = metropolis_hastings(target_distribution, proposal_distribution, initial_sample, iterations, num_samples, step_size)




import sys
import os

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.dirname(current_dir)

# Add the parent directory to the sys.path
sys.path.append(parent_dir)

# Now you can import the function from the analysis module
from analysis import sample_distribution_history


sample_distribution_history(samples_history, 'example', 1)