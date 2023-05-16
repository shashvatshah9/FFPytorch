from torch import Tensor


class FFEncoding(object):
    @staticmethod
    def overlay(x: Tensor, y: Tensor):
        """
        Replace the first 10 pixels of data [x] with one-hot-encoded label [y]
        """
        x_ = x.clone()
        x_[:, :10] *= 0.0
        x_[range(x.shape[0]), y] = x.max()
        return x_

    @staticmethod
    def overlay2d(x: Tensor, y: Tensor):
        """
        Replace the first 10 pixels of data [x] for all channels with one-hot-encoded label [y]
        """
        assert x[0].size(dim=0) == 3, "Expects a 3 channel image"
        x_ = x.clone()
        x_[:, :, :10] *= 0.0
        x_[range(x.size(dim=0)), range(x.size(dim=1)), y] = x.max()
        return x_
