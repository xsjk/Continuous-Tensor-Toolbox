import torch
from torch import nn
import itertools


def soft_thres(x, lam):
    return torch.sign(x) * torch.relu(torch.abs(x) - lam)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0, bias=True):
        super().__init__()
        self.omega_0 = omega_0
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.__init_weights()

    @torch.no_grad()
    def __init_weights(self):
        stdv = 1 / self.in_features
        # stdv = (6 / self.in_features) ** 0.5 / self.omega_0
        self.linear.weight.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.uniform_(-stdv, stdv)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class SineSequential(nn.Sequential):
    def __init__(self, sizes: list[int], omega_0):
        assert len(sizes) >= 2, "Need at least two sizes for input and output"
        super(SineSequential, self).__init__(
            *(
                SineLayer(sizes[i], sizes[i + 1], omega_0)
                for i in range(len(sizes) - 2)
            ),
            nn.Linear(sizes[-2], sizes[-1]),
        )


class LRTF(nn.Module):
    def __init__(
        self, kernel_size: tuple[int, ...], hidden_size: tuple[int], omega_0: float
    ):
        """Low-Rank Tensor Factorization (LRTF) model.

        Parameters
        ----------
        kernel_size : tuple[int, ...]
            The size of the kernel.
        hidden_size : tuple[int]
            The size of the hidden layers in the SineSequential.
        omega_0 : float
            The base frequency of the sine function.

        Examples
        --------
        >>> model = LRTF(kernel_size=(3, 3), hidden_size=[3], omega_0=2)
        >>> model
        LRTF(
          (f): ModuleList(
            (0-1): 2 x SineSequential(
              (0): SineLayer(
                (linear): Linear(in_features=1, out_features=3, bias=True)
              )
              (1): Linear(in_features=3, out_features=3, bias=True)
            )
          )
        )
        >>> a = torch.randn(10)
        >>> b = torch.randn(20)
        >>> c = torch.randn(10)
        >>> model(a, b).shape
        torch.Size([10, 20])
        >>> model.diag(a, c).shape
        torch.Size([10])
        """

        super(LRTF, self).__init__()
        self.C = nn.Parameter(
            torch.empty(kernel_size).uniform_(-(stdv := kernel_size[0] ** -0.5), stdv)
        )
        self.f = nn.ModuleList(
            [SineSequential([1, *hidden_size, r], omega_0) for r in kernel_size]
        )

    def forward(self, *inputs: torch.Tensor):
        n = self.C.ndim
        assert len(inputs) == n, "Number of inputs must match the kernel ndim"
        assert all(x.ndim == 1 for x in inputs), "All inputs must be 1D"
        return torch.einsum(
            self.C,
            torch.arange(n),
            *itertools.chain(
                *(
                    (f(x.unsqueeze(1)), [i + n, i])
                    for i, f, x in zip(range(n), self.f, inputs)
                )
            ),
            torch.arange(n, 2 * n),
        )
        X = self.C
        for f, x in zip(self.f, inputs):
            X = torch.tensordot(X, f(x), dims=([0], [1]))
        return X

    def diag(self, *inputs: torch.Tensor):
        '''
        Directly compute the diagonal of the tensor.
        
        Equal to the forward method with the same inputs, but only returns the diagonal.

        Parameters
        ----------
        inputs : torch.Tensor
            The input tensors.

        Returns
        -------
        torch.Tensor
            The diagonal of the tensor.
        '''
        n = self.C.ndim
        assert len(inputs) == n, "Number of inputs must match the kernel ndim"
        assert all(x.ndim == 1 for x in inputs), "All inputs must be 1D"
        assert len(set(map(len, inputs))) == 1, "All inputs must have the same length"
        return torch.einsum(
            self.C,
            torch.arange(n),
            *itertools.chain(
                *(
                    (f(x.unsqueeze(1)), [n, i])
                    for i, f, x in zip(range(n), self.f, inputs)
                )
            ),
            [n],
        )
