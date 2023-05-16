import matplotlib.pyplot as plt
import torch
import time
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm

from FFNetwork import FFNetworkBatched
from FFEncoding import FFEncoding

overlay_y_on_x = FFEncoding.overlay



def get_data_loaders(train_batch_size=50, test_batch_size=50):
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


def visualize_sample(data, name="", idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


def training_loop(model, iterator, device, encoding="overlay"):
    print("Training...")
    model.train()
    model(iterator, device)


def test_loop(model, test_loader, device):
    print("Evaluating...")
    model.eval()
    batch_error = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        batch_error += calc_error(model, x, y, device)

    avg_error = batch_error / len(test_loader)
    print(f"error: {avg_error}")
    return avg_error

def train_loop(model, train_loader, device):
    print("Evaluating TRAIN...")
    model.eval()
    batch_error = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        batch_error += calc_error(model, x, y, device)

    avg_error = batch_error / len(train_loader)
    print(f"error: {avg_error}")
    return avg_error

def eval_loop(model, input, device, batched_per_layer=False, encoding="overlay"):
    if batched_per_layer == True:
        if encoding == "overlay":
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(input, label)
                goodness = []
                for module in model.children():
                    module.to(device)
                    h = module(h)
                    goodness += [h.pow(2).mean(1)]
                    module.to("cpu")
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label.argmax(1)
    else:
        model.to(device)
        if encoding == "overlay":
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(input, label)
                goodness = []
                for module in model.children():
                    h = module(h)
                    goodness += [h.pow(2).mean(1)]
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label.argmax(1)


def calc_error(model, x, y, device) -> float:
    model.eval()
    return 1 - eval_loop(model, x, device).eq(y).float().mean().item()


if __name__ == "__main__":
    # Define parameters
    EPOCHS = 40
    BATCH_SIZE = 50
    TRAIN_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    encoding = "overlay"
    torch.manual_seed(1234)

    # Build train and test loaders
    train_loader, test_loader = get_data_loaders(
        train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE
    )

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build network
    net = FFNetworkBatched([784, 500, 500])

    # Iterator in place of DataLoader
    data_iter = []

    # Encode true and false labels on images to create positive and negative data
    print("Encoding positive and negative data with correct and incorrect labels")
    for x, y in tqdm(train_loader):
        x_pos, x_neg = None, None
        if encoding == "overlay":
            x_pos = overlay_y_on_x(x, y)
            rand_mask = torch.randint(0, 9, y.size())
            y_rnd = (y + rand_mask + 1) % 10
            x_neg = overlay_y_on_x(x, y_rnd)

        data_iter.append((x_pos, x_neg))
    training_errors = []
    testing_errors = []
    # Train / test
    for epoch in range(EPOCHS):
        print(f"==== EPOCH: {epoch} ====")
        start = time.time()
        training_loop(net, data_iter, device)
        training_error = train_loop(net, train_loader, device)
        print("Training.....")        
        training_errors.append(training_error)

        testing_error = test_loop(net, test_loader, device)
        testing_errors.append(testing_error)
        end = time.time()
        elapsed = end - start
        print(f"Completed epoch {epoch} in {elapsed} seconds")
    # Plot errors
    plt.figure(figsize=(10, 6))  # Adjust the figsize to make the plot wider
    plt.plot(range(1, EPOCHS + 1), training_errors, label='Training Error')  # Use training_errors instead of errors
    plt.plot(range(1, EPOCHS + 1), testing_errors, label='Testing Error')  # Use testing_errors instead of errors
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error over Epochs')
    plt.legend()
    plt.xticks(range(1, EPOCHS + 1))  # Set x-axis tick labels to 1, 2, ...
    plt.ylim(0.0, 0.13)  # Adjust the y-axis limits to zoom in on the range of errors
    plt.savefig('error_plot_30_epochs.png')  # Save the plot as a PNG file
    plt.show()
