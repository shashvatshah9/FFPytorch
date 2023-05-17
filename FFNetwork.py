from typing import Iterator
import torch
import copy
from torch.nn import Module
from torch._jit_internal import _copy_to_script_wrapper
from FFLayer import FFLayer


class FFNetwork(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) >= 1, "len(dims) should be greater than equal to 1"
        for d in range(len(dims) - 1):
            self.add_module(str(d), FFLayer(dims[d], dims[d + 1], torch.nn.GELU()))

    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, *input):
        if self.training:
            # assert len(input) == 2, "Pass both positive and negative input"
            x_pos, x_neg = input[0]
            device = input[1]
            x_pos, x_neg = x_pos.to(device), x_neg.to(device)
            for i, module in enumerate(self.children()):
                # print("Training layer", i, "...")
                x_pos, x_neg = module(x_pos, x_neg)
            return
        else:
            assert len(input) == 1, "Only pass the input data "
            for module in self:
                input = module(input)
            return input


class FFNetworkBatched(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        assert len(dims) >= 1, "len(dims) should be greater than equal to 1"
        for d in range(len(dims) - 1):
            self.add_module(str(d), FFLayer(dims[d], dims[d + 1], torch.nn.GELU()))

    def __len__(self) -> int:
        return len(self._modules)

    @_copy_to_script_wrapper
    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    @_copy_to_script_wrapper
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def forward(self, input, device):
        if self.training:
            # Loop through all batches in dataset
            layer_data = copy.deepcopy(input)
            # Loop through each layer
            for i, module in enumerate(self.children()):
                print(f"Training layer {i}")
                module.to(device)
                module.zero_grad()
                for i, x_data in enumerate(layer_data):
                    x_pos, x_neg = x_data[0], x_data[1]
                    x_pos, x_neg = x_pos.to(device), x_neg.to(device)
                    x_pos, x_neg = module(x_pos, x_neg)
                    x_pos = x_pos.detach()
                    x_neg = x_neg.detach()
                    x_pos, x_neg = x_pos.to("cpu"), x_neg.to("cpu")
                    layer_data[i] = (x_pos, x_neg)
                module.to("cpu")
            return
        else:
            assert len(input) == 1, "Only pass the input data "
            for module in self:
                input = module(input)
            return input
