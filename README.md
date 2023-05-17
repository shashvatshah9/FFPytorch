# FFPytorch
Forward Forward Implementation improvements

References
1. https://github.com/mohammadpz/pytorch_forward_forward
2. [Geoffrey Hinton's talk at NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf).

Code Structure
FFEncoding -  provides methods to overlay one-hot-encoded labels onto input data, either in the form of replacing the first 10 pixels of the data with the label or replacing the first 10 pixels in each channel of a 2D image with the corresponding label.
FFLayer - defines a feed-forward layer (FFLayer) that extends the Linear class from PyTorch. It takes input features, output features, an activation function, an optimizer, a threshold value, and other parameters. 
FFNetwork - defines two classes: FFNetwork and FFNetworkBatched. FFNetwork is a feed-forward neural network module that sequentially passes inputs through layers, while FFNetworkBatched supports batched processing of inputs and applies the forward pass to each batch.

How to run? 
Assuming you have a python-supporting environment, which has available gpu(s), you can type this in the terminal *python base.py*.
