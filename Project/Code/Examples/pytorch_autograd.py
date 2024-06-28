import torch

x = torch.tensor(2., requires_grad=True)
y = x**3

first = torch.autograd.grad(y, x, create_graph=True)[0]
second = torch.autograd.grad(first, x)[0]

print(f'First derivative of f({x}):', first.item())
print(f'Second derivative of f({x}):', second.item())


x = torch.tensor(3., requires_grad=True)
y = torch.tensor(3., requires_grad=True)

func = 5*x**4 - 5*y*x**2 + 3*y**3 - 55*y + x*y - 3

first_x = torch.autograd.grad(func, x, create_graph=True)[0]
first_y = torch.autograd.grad(func, y, create_graph=True)[0]

second_x = torch.autograd.grad(first_x, x)[0]
second_y = torch.autograd.grad(first_y, y)[0]

print(f'First derivative of f({x}, {y}):', first_x.item(), first_y.item())
print(f'Second derivative of f({x}, {y}):', second_x.item(), second_y.item())