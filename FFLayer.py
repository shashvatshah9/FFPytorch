import torch
from torch.nn import Module, Linear, ReLU
from tqdm import tqdm
from torch.optim import Adam, Optimizer


class FFLayer(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Module = ReLU,
        optimizer: Optimizer = None,
        threshold: float = 2.0,
        layer_iters: int = 10,
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
        self.layer_iters = layer_iters

    def forward_pass(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.activation(
            torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0)
        )

    def forward(self, *input):
        if self.training:
            assert len(input) == 2, "Pass both positive and negative input"

            x_pos = input[0]
            x_neg = input[1]

            for _ in range(self.layer_iters):
                g_pos = self.forward_pass(x_pos).pow(2).mean(1)
                g_neg = self.forward_pass(x_neg).pow(2).mean(1)
                # The following loss pushes pos (neg) samples to
                # values larger (smaller) than the self.threshold.
                loss = torch.log(
                    1
                    + torch.exp(
                        torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
                    )
                ).mean()
                self.opt.zero_grad()
                # this backward just compute the derivative and hence
                # is not considered backpropagation.
                loss.backward(retain_graph=True)
                self.opt.step()
            x1, x2 = self.forward_pass(x_pos), self.forward_pass(x_neg)
            return x1, x2
        else:
            assert len(input) == 1, "Pass only 1 argument in eval mode"
            r = self.forward_pass(input[0])
            return r
