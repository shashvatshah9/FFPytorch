import torch
from torch.nn import Module, ModuleList
from torch.optim import Adam, Optimizer
import torch.optim as optim
from typing import Iterator
import tqdm

class FFNetwork(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) >= 1, "len(dims) should be greater than equal to 1"
        
        layers = []
        for d in range(len(dims) - 1):
            layers.append(nn.Linear(dims[d], dims[d+1]))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def forward(self, input):
        return self.layers(input)

    def predict(self, x):
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, 1)
            return predicted
        
        
# create an instance of the FFNetwork model
ffnet = FFNetwork([784, 256, 256, 10])

# set up the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ffnet.parameters(), lr=0.001)

# train the model for num_epochs epochs
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(train_loader, 0):
        # assume data is loaded in train_loader
        # enumerate over batches
        # torch.max - get predictions
        # prints metrics
        inputs, labels = data
        optimizer.zero_grad()
        outputs = ffnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    print(f"Epoch {epoch + 1} - Loss: {epoch_loss:.3f} - Accuracy: {epoch_accuracy:.2f}%")
