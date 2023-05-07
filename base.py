import matplotlib.pyplot as plt
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm

from FFNetwork import FFNetwork
from FFEncoding import FFEncoding

overlay_y_on_x = FFEncoding.overlay


def MNIST_loaders(train_batch_size=1000, test_batch_size=1000):
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


# def training_loop(model, x, y, encoding="overlay"):
#     x_pos, x_neg = None, None
#     if encoding == "overlay":
#         x_pos = overlay_y_on_x(x, y)
#         rand_mask = torch.randint(0, 9, y.size()).to(device)
#         y_rnd = (y + rand_mask + 1) % 10
#         x_neg = overlay_y_on_x(x, y_rnd)
#     model.train()
#     model(x_pos, x_neg)


def training_loop(model, train_loader, device, encoding="overlay"):
    print("Training loop")
    model.to(device)
    for i, (x, y) in enumerate(train_loader):
        print(f"\nBatch {i} out of {len(train_loader)}")
        x_pos, x_neg = None, None
        if encoding == "overlay":
            x_pos = overlay_y_on_x(x, y)
            rand_mask = torch.randint(0, 9, y.size())
            y_rnd = (y + rand_mask + 1) % 10
            x_neg = overlay_y_on_x(x, y_rnd)
        x_pos, x_neg = x_pos.to(device), x_neg.to(device)
        # model.to(device)
        model.train()
        model(x_pos, x_neg)
        # model.to('cpu')
        error = calc_error(model, x, y, device)
        print(f"Error {error}")
    model.to('cpu')



def eval_loop(model, x, encoding="overlay"):
    if encoding == "overlay":
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for module in model.children():
                h = h.to(device)
                # module.to(device)
                h = module(h)
                # module.to('cpu')
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        # print(f"goodness: {goodness_per_label}")
        return goodness_per_label.argmax(1)


# def eval_loop(model, train_loader, encoding="overlay"):
#     if encoding == "overlay":
#         for x, _ in train_loader:
#             for label in range(10):
#                 h = overlay_y_on_x(x, label)
#                 goodness = []
#                 for module in model.children():
#                     h = module(h)
#                     goodness += h.pow(2).mean(1)
                
            

def test_loop(model, test_loader, device):
    model.eval()
    batch_error = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        batch_error += calc_error(model, x, y)
    
    avg_error = batch_error / len(test_loader)
    print(avg_error)

def calc_error(model, x, y, device) -> float:
    model.eval()
    r1 = eval_loop(model, x)
    r1 = r1.to('cuda')
    y = y.to('cuda')
    r2 = r1.eq(y).float().mean().item()
    r = 1 - r2
    return r


torch.manual_seed(1234)
train_loader, test_loader = MNIST_loaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = FFNetwork([784, 500, 500])
# net = net.to(device)
# x, y = next(iter(train_loader))
# x, y = x.to(device), y.to(device)

# training_loop(net, x, y)
training_loop(net, train_loader, device)
# net.eval()
# print(calc_error(net, x, y))

# x_te, y_te = next(iter(test_loader))
# x_te, y_te = x_te.cuda(), y_te.cuda()

# USE EVAL
# net.eval()
# print(calc_error(net, x_te, y_te))
test_loop(net, test_loader, device)
