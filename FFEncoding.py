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
