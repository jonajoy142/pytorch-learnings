import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ---------------------------
# DATA
# ---------------------------
transform = transforms.ToTensor()

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------------------
# MODEL (DYNAMIC)
# ---------------------------
model = nn.Sequential(

    # Input: (1, 28, 28)

    nn.Conv2d(1, 32, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),

    nn.LazyLinear(128),   # no manual calculation
    nn.ReLU(),
    nn.Linear(128, 10)
)

# ---------------------------
# LOSS + OPTIMIZER
# ---------------------------
loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ---------------------------
# TRAINING
# ---------------------------
for epoch in range(5):

    total_loss = 0

    for images, labels in train_loader:

        predictions = model(images)

        loss = loss_fn(predictions, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print("epoch:", epoch, "loss:", avg_loss)

# ---------------------------
# TESTING
# ---------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:

        outputs = model(images)
        predicted = outputs.argmax(1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("accuracy:", correct / total)