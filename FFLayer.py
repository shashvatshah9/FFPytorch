import torch
from torch.nn import Module, Linear, ReLU
from tqdm import tqdm
from torch.optim import Adam, Optimizer


class Layer(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Module = ReLU(),
        optimizer: Optimizer = None,
        threshold: float = 2.0,
        num_epochs: int = 1000,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.activation = activation
        if optimizer == None:
            self.opt = Adam(self.parameters(), lr=0.03)
        else:
            self.opt = optimizer
        self.threshold = threshold
        self.num_epochs = num_epochs

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        output = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        if isinstance(self.activation, ReLU):
            output = torch.square(output)
        else:
            output = self.activation(output)
        return output

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(
                1
                + torch.exp(
                    torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                )
            ).mean()
            self.opt.zero_grad()
            # this backward just computes the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

    def predict(self, x):
      with torch.no_grad():
          x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
          output = torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
          if isinstance(self.activation, ReLU):
              output = torch.square(output)
          else:
              output = self.activation(output)
          return output.argmax(dim=1)
