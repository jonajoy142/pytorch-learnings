import torch

# training data
x = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[2.0],[4.0],[6.0],[8.0]])

# model (1 input -> 1 output)
model = torch.nn.Linear(1,1)

# loss function
loss_fn = torch.nn.MSELoss()

# optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training loop
for epoch in range(310):

    pred = model(x)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("weight learned:", model.weight.item())