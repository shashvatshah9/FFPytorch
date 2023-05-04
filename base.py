import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader

from FFNetwork import FFNetwork
import FFEncoding

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

    h_pos, h_neg = x_pos, x_neg
    # output = model(x_pos, x_neg, encoding)
    for i, module in enumerate(model.children()):
        print("Training layer", i, "...")
        h_pos, h_neg = module.train(h_pos, h_neg)


torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders()

net = FFNetwork([784, 500, 500])
net = net.cuda()
x, y = next(iter(train_loader))
x, y = x.cuda(), y.cuda()

training_loop(net, x, y)

print("train error:", 1.0 - net.predict(x).eq(y).float().mean().item())

x_te, y_te = next(iter(test_loader))
x_te, y_te = x_te.cuda(), y_te.cuda()

print("test error:", 1.0 - net.predict(x_te).eq(y_te).float().mean().item())
