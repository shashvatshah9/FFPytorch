import matplotlib.pyplot as plt
import torch
import time
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt



class BackpropNetwork(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            # self.layers.append(self.linear_layer(dims[d], dims[d+1]))
            self.add_module(str(d), self.linear_layer(dims[d], dims[d + 1]))

    def linear_layer(self, in_dim, out_dim):
        layer = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True)
        )
        return layer
    
    def forward(self, x):
        for i, module in enumerate(self.children()):
            x = module(x)
        return x
    
    
def MNIST_loaders(train_batch_size=50, test_batch_size=50):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_loader = DataLoader(
        MNIST("./data/", train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        MNIST("./data/", train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def training_loop(model, iterator, loss_fn, optimizer, device):
    epoch_loss = 0.0
    epoch_err = 0.0
    model.train()

    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        error = 1 - calculate_accuracy(y_hat, y).item()

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_err += error
        
    return epoch_loss / len(iterator), epoch_err / len(iterator)


def test_loop(model, iterator, loss_fn, device):
    epoch_loss = 0.0
    epoch_err = 0.0
    model.eval()
    
    with torch.no_grad():
        for (x, y) in iterator:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            error = 1 - calculate_accuracy(y_pred, y).item()

            epoch_loss += loss.item()
            epoch_err += error
        
    return epoch_loss / len(iterator), epoch_err / len(iterator)



if __name__ == "__main__":
    # Define parameters
    EPOCHS = 20
    BATCH_SIZE=50
    TRAIN_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    torch.manual_seed(1234)

    # Build train and test loaders
    train_loader, test_loader = MNIST_loaders(train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build network
    model = BackpropNetwork([784, 500, 10])
    model = model.to(device)
    print(summary(model, (1, 784)))

    # Define loss function and optimizer
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Record 
    test_losses = []
    test_errors = []
    train_losses = []
    train_errors = []

    # Train / test
    for epoch in range(EPOCHS):
        print(f"==== EPOCH: {epoch} ====")
        start = time.time()
        train_loss, train_err = training_loop(model, train_loader, loss, optimizer, device)
        train_losses.append(train_loss)
        train_errors.append(train_err)
        test_loss, test_err = test_loop(model, test_loader, loss, device)
        test_losses.append(test_loss)
        test_errors.append(test_err)
        end = time.time()
        elapsed = end - start
        print(f"train loss: {train_loss} / error: {train_err}    test loss: {test_loss} / error: {test_err}")
        print(f"Completed epoch {epoch} in {elapsed} seconds")

    fig, axs = plt.subplots(2, 1)

    axs[0].plot(range(len(train_losses)), train_losses, label='train loss', color='green')
    axs[0].plot(range(len(test_losses)), test_losses, label='test loss', color='red')
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].legend()
    
    axs[1].plot(range(len(train_errors)), train_errors, label='train err', color='green')
    axs[1].plot(range(len(test_errors)), test_errors, label='test err', color='red')
    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('error')
    axs[1].legend()


    plt.tight_layout()
    plt.show()