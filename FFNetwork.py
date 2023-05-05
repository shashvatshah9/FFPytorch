import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from FFLayer import *
import FFEncoding

# define the single-layer neural network
net = Layer(784, 10)

# load the MNIST dataset
train_data = MNIST(root="./data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# train the network on the MNIST dataset
for images, labels in train_loader:
    # convert images and labels to the appropriate format
    images = images.view(images.shape[0], -1)
    labels = labels.type(torch.float32)

    # create positive and negative samples
    pos_idx = (labels == 1)
    neg_idx = (labels == 0)
    x_pos = images[pos_idx]
    x_neg = images[neg_idx]

    # train the network for one epoch
    net.train(x_pos, x_neg)

# test the network on the MNIST test set
test_data = MNIST(root="./data", train=False, download=True, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

correct = 0
total = 0

for images, labels in test_loader:
    # convert images to the appropriate format
    images = images.view(images.shape[0], -1)

    # compute the network's predictions
    predictions = net.predict(images)

    # update the accuracy count
    correct += (predictions == labels).sum()
    total += labels.shape[0]

accuracy = (float(correct) / float(total))
print("Accuracy % :", accuracy)
