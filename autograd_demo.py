import torch

x = torch.tensor(2.0, requires_grad=True)

y = x**2 + 3*x

print("y:", y)

y.backward()

print("gradient:", x.grad)