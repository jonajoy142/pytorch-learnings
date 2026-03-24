
import torch

X = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y = torch.tensor([[4.0],[7.0],[10.0],[13.0]])


model = torch.nn.Linear(1,1)

loss_fn = torch.nn.MSELoss()


optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# training loop
for epoch in range(210):

    pred = model(X)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("weight:", model.weight.item())
    print("loss:", loss.item())

print("weight learned:", model.weight.item())