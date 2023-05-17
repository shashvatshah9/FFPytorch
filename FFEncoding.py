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
        assert x.size(dim=1) % 3 == 0, "Expects a 3 channel image"
        img_size = int(x.size(dim=1) / 3)
        x_ = x.clone()
        x_[:, 0:10] *= 0.0
        x_[range(x.shape[0]), y] = x.max()

        x_[:, img_size : img_size + 10] *= 0.0
        x_[range(x.shape[0]), img_size + y] = x.max()

        x_[:, img_size * 2 : img_size * 2 + 10] *= 0.0
        x_[range(x.shape[0]), img_size * 2 + y] = x.max()

        return x_
