import torch
import torch.optim as optim
import numpy as np

from wavefunctions import Wavefunctions

def trial_wavefunction(name):
    init = Wavefunctions(name)
    return init.wf

def loss_fn(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def pre_train_NN(model, name, num_particles, dof):
    lattice = ['ising', 'heisenberg']
    
    # Generate initial samples
    x = [torch.randn(100, dof) for _ in range(num_particles)]
    x = torch.cat(x, dim=-1)
    
    if name in lattice:
        x = np.where(np.random.uniform(size=(1000, num_particles)) > 0.5, 0.5, -0.5)
        x = torch.tensor(x, dtype=torch.float32)
    elif name == 'two_fermions':
        x = [torch.randn(100, dof) for _ in range(num_particles)]
        x = torch.cat(x, dim=-1)
    
    optimizer = optim.Adam(model.NN.parameters())

    y_values = torch.tensor(trial_wavefunction(name)(x), dtype=torch.float32)

    for epoch in range(500):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_values, y_pred)
        loss.backward()
        optimizer.step()

    torch.save(model.NN.state_dict(), f'./weights/{name}_weights.pth')

