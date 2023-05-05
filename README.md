# FFPytorch
Forward Forward Implementation improvements

References
1. https://github.com/mohammadpz/pytorch_forward_forward
2. [Geoffrey Hinton's talk at NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf).



# Comments
************************************************************************
Using Forward Forward for classification tasks trained on MNIST:

One approach might be to treate each pixel of the image as a discrete observation, and consider the whole image as a sequence of observations (similar to time-series analysis). 

What does the code do : 

1. Class FFEncoding - preprocess step: replace the first 10 pixels with a 1-hot encoded version of the label. 
2. FFLayer - use loss to learn embeddings that distinguish b/n + and - samples based on a threshold. 



Now, the most straightforward idea, is to use a Multi-Layer-Perceptron architecture. It consists several FC layers with non-linear activation functions between them. We use Cross Entropy Loss and Adam optimizer. 
@todo check out different hidden_dim values for potential optimization.
Some might include changes in Network Depth (assuming there is a direct relationship b/n performance and depth); try out different activation functions (leaky relu); try to avoid overfitting by using regularization;   



# Note: We can replace backprop by doing forword algorithm for the MLP. Not sure if it is valid, but here it is : 
 # Forward algorithm
        alpha = outputs[0].unsqueeze(0)
        for j in range(1, len(labels)):
            alpha_j = alpha.mm(mlp.fc2.weight).add(mlp.fc2.bias).relu()
            alpha = torch.cat((alpha, alpha_j), dim=0)
            alpha[-1] = nn.functional.softmax(alpha[-1], dim=1)
            
        loss = -torch.log(alpha[-1, labels[-1]])
        running_loss += loss.item() 
        
        
# Architecture summary 

This code implements a simple training process for a single layer neural network without backpropagation. The neural network is defined using the PyTorch Linear module and an activation function. The training process uses the Adam optimizer and a custom loss function to update the weights of the neural network.

The forward method of the Layer class takes an input tensor and returns the output tensor of the neural network after applying the activation function. The input tensor is first normalized and then multiplied by the weights of the layer, and the bias term is added.

The train method of the Layer class takes two input tensors, x_pos and x_neg, representing positive and negative samples, respectively. The training process uses the forward method to compute the output tensors for both positive and negative samples, and then computes the loss function using the output tensors and the threshold value. The loss is then used to update the weights of the neural network using the Adam optimizer. The training process is repeated for a fixed number of epochs, and the final output tensors for the positive and negative samples are returned.

Note that this training process does not use backpropagation to compute the gradients of the loss function with respect to the weights of the neural network. Instead, it computes the derivative of the loss function with respect to the output tensors and uses this to update the weights directly. This approach is often called "derivative-free" or "implicit" learning and can be useful in some scenarios where backpropagation is not feasible or desirable.
