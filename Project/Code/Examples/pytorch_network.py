import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class NeuralNetwork(nn.Module):
    def __init__(self, N):
        super(NeuralNetwork, self).__init__()
        self.N = N
        self.model = nn.Sequential(
            nn.Linear(self.N, 25),
            nn.ReLU(),
            nn.Linear(25, self.N),
        )
 
    def forward(self, x):
        return self.model(x)
    
def targetfunc(x):
    return x**3 - 5*x + 3 

N = 50

x = torch.linspace(-5, 5, N)

y_values = targetfunc(x)

NN = NeuralNetwork(N)
MSE = nn.MSELoss()

optimizer = torch.optim.Adam(NN.model.parameters())

pytorch_total_params = sum(p.numel() for p in NN.model.parameters() if p.requires_grad)
print('Total trainable parameters:', pytorch_total_params)

for epoch in range(500):
    optimizer.zero_grad()
    y_pred = NN(x)
    loss = MSE(y_values, y_pred)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(loss.item())
    
x = x.detach()
y_pred = y_pred.detach()
y_values = y_values.detach()
plt.plot(x, y_values, 'k')
plt.plot(x, y_pred, '-.r')
plt.legend(['True', 'Predicted'])
plt.show()
    
