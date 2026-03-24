import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
transform = transforms.ToTensor()
import torch.nn as nn

train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

    # model = nn.Sequential(
    #     nn.Flatten(),
    #     nn.Linear(28*28, 10)
    # )
# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(784,128),
#     nn.ReLU(),
#     nn.Linear(128,10)
# )

# model = nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(784,512),
#     nn.ReLU(),
#     nn.Linear(512,265),
#     nn.ReLU(),
#     nn.Linear(256, 128),
#     nn.ReLU(),
#     nn.Linear(128, 10)
# )

model = nn.Sequential(
    # Input: (1, 28, 28)

    nn.Conv2d(1, 32, kernel_size=3),  # (16, 26, 26)
    nn.ReLU(),
    nn.MaxPool2d(2),                   # (16, 13, 13)

    nn.Conv2d(32, 64, kernel_size=3),  # (32, 11, 11)
    nn.ReLU(),
    nn.MaxPool2d(2),                   # (32, 5, 5)

    nn.Flatten(),                      # (32*5*5 = 800)


    nn.Linear(64 * 5 * 5, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# nn.Conv2d(1, 16, 3)
# nn.Conv2d(16, 32, 3)
# nn.Linear(32 * 5 * 5, 128),

# nn.Conv2d(1, 32, 3)
# nn.Conv2d(32, 64, 3)
# nn.Linear(64 * 5 * 5, 128),


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

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

correct = 0
total = 0

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        predicted = outputs.argmax(1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("accuracy:", correct / total)