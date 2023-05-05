import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from FFNetwork import FFNetwork
import FFEncoding

overlay_y_on_x = FFEncoding.overlay

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

overlay_y_on_x = FFEncoding.overlay


def MNIST_loaders(train_batch_size=50000, test_batch_size=10000):
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


def training_loop(model, x, y, encoding="overlay"):
    x_pos, x_neg = None, None
    if encoding == "overlay":
        x_pos = overlay_y_on_x(x, y)
        rnd = torch.randperm(x.size(0))
        x_neg = overlay_y_on_x(x, y[rnd])

    model.train()
    model(x_pos, x_neg)


def eval_loop(model, x, encoding="overlay"):
    if encoding == "overlay":
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for module in model.children():
                h = module(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)


def calc_error(model, x, y) -> float:
    model.eval()
    return 1 - eval_loop(model, x).eq(y).float().mean().item()


torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FFNetwork([784, 500, 500])
net = net.to(device)
x, y = next(iter(train_loader))
x, y = x.to(device), y.to(device)

training_loop(net, x, y)
net.eval()
print(calc_error(net, x, y))

x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.cuda(), y_te.cuda()

# USE EVAL
net.eval()
print(calc_error(net, x_te, y_te))
