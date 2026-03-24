import torch
import torch.nn as nn
import torch.optim as optim

# XOR dataset
X = torch.tensor([
    [0.,0.],
    [0.,1.],
    [1.,0.],
    [1.,1.]
])

y = torch.tensor([
    [0.],
    [1.],
    [1.],
    [0.]
])

# neural network
class XORModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(2,4),
            nn.ReLU(),
            nn.Linear(4,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.net(x)


model = XORModel()

loss_fn = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.1)

# training
for epoch in range(2000):

    pred = model(X)

    loss = loss_fn(pred, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 200 == 0:
        print(epoch, loss.item())


# test model
print("\nPredictions:")

print(model(X))