import torch
import torch.nn as nn


class Quadratic(nn.Module):
    """Compute ``X A X^T``.

    Attributes
    ----------
    A: Tensor(n x n)
        Defaults to the identity vector

    Examples
    --------

    >>> q = Quadratic((10, 20))
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = q(x)
    >>> result.shape
    torch.Size([3, 10, 10])
    """
    def __init__(self, input_shape):
        super(Quadratic, self).__init__()
        _, m = input_shape
        self.A = nn.Parameter(torch.eye(m, dtype=torch.float32))

    def forward(self, x):
        """

        Parameters
        ----------
        x: N x S x T

        Returns
        -------
        A tensor of shape (N x S x S)
        """
        xt = x.transpose(2, 1)
        r1 = torch.matmul(x, self.A)
        return torch.matmul(r1, xt)


class Differential(nn.Module):
    """Compute a 1d convolution with a default filter of [1, -1]. It is used to compute
    log returns in finance.

    Parameters
    ----------
    out_channel: int
        Number of output, if != input it means the network will create buckets of assets

    Notes
    -----
    Recent timesteps are last

    Examples
    --------

    >>> diff = Differential((10, 20), k=3)
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = diff(x)
    >>> result.shape
    torch.Size([3, 10, 18])
    >>> torch.abs(result[0, 0, 0] - (x[0, 0, 2] - x[0, 0, 1])) < 1e-4
    tensor(True)
    """
    def __init__(self, input_shape, k=2, out_channel=None):
        super(Differential, self).__init__()
        n, _ = input_shape

        filter = torch.tensor([0 for _ in range(k)]).float()
        filter[-2] = -1
        filter[-1] = 1

        if out_channel is None:
            out_channel = n

        self.out_channel = out_channel
        in_channel = n
        kernel = torch.randn(out_channel, in_channel, k).float()
        kernel[:] = 0

        for i in range(n):
            kernel[i, i, :] = filter

        self.kernel = nn.Parameter(kernel, requires_grad=True)
        # print(self.kernel)

    def forward(self, x):
        """
        Parameters
        ----------
        x: N x S x T

        Returns
        -------
        A tensor of shape (N x S x (T - (k - 1)))
        """
        return nn.functional.conv1d(x, self.kernel)


class Barycenter(nn.Module):
    """Compute the barycenter of a sample.

    Attributes
    ----------
    weight: Tensor(T x 1)
        Defaults to 1 / T (equivalent to the mean)

    Examples
    --------

    >>> barycenter = Barycenter((10, 20))
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = barycenter(x)
    >>> result.shape
    torch.Size([3, 10, 1])
    >>> torch.abs(result - x.mean(dim=2).unsqueeze(2)).sum() < 1e-5
    tensor(True)

    """
    def __init__(self, input_shape):
        super(Barycenter, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_shape[1], 1, dtype=torch.float32) / input_shape[1])

    def forward(self, x):
        return torch.matmul(x, self.weight / self.weight.sum())


class Covariance(nn.Module):
    """Compute the covariance matrix

    Parameters
    ----------
    input_shape: (S, T)
        Shape of the expected tensor (batch size should not be in the shape)

    k: int
        Number of time steps used to compute the returns (with k=2, return=x_t - x_(t - 1))

    bias: bool
        If true use T to compute the covariance, else use T - 1

    Notes
    -----
    The initialization defaults the standard covariance formula.
    During training the weights change in unpredictable ways.

    Examples
    --------
    >>> cov = Covariance((10, 20), k=3, bias=True)
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = cov(x)
    >>> result.shape
    torch.Size([3, 10, 10])
    >>> import numpy as np
    >>> ret = (x[0, :, 1:] - x[0, :, 0:-1]).numpy()
    >>> numpy_cov = np.cov(ret, bias=True)
    >>> torch.abs(result[0, :, :] - torch.from_numpy(numpy_cov)).sum() < 1e-5
    tensor(True)


    >>> cov = Covariance((10, 20), k=3, channels=100, bias=True)
    >>> batch_size = 3
    >>> x = torch.randn((batch_size, 10, 20))
    >>> result = cov(x)
    >>> result.shape
    torch.Size([3, 100, 100])
    """
    def __init__(self, input_shape, k=2, channels=None, bias=True):
        super(Covariance, self).__init__()

        n, m = input_shape

        self.diff = Differential((n, m), k=k, out_channel=channels)   # x_t - x_{t - 1}
        n = self.diff.out_channel

        self.quadratic = Quadratic((n, m - (k - 1)))  # r * r^T
        self.mean = Barycenter((n, m - (k - 1)))      # r * 1

        if bias:
            self.m = m - 1
        else:
            self.m = m - 2

    def forward(self, x):
        returns = self.diff(x)

        squared = self.quadratic(returns)
        mean    = self.mean(returns)

        return squared / self.m - torch.matmul(mean, mean.transpose(2, 1))
