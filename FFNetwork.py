from typing import Iterator
import torch
from torch.nn import Module
from torch._jit_internal import _copy_to_script_wrapper
import FFLayer
import FFEncoding

overlay_y_on_x = FFEncoding.overlay


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
            assert len(input) == 2, "Pass both positive and negative input"
            x_pos, x_neg = tuple(input)
            for i, module in enumerate(self.children()):
                print("Training layer", i, "...")
                x_pos, x_neg = module(x_pos, x_neg)
            return
        else:
            assert len(input) == 1, "Only pass the input data "
            for module in self:
                input = module(input)
            return input
