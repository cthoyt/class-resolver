# -*- coding: utf-8 -*-

"""Class resolvers for PyTorch."""

from torch import nn
from torch.optim import Adam, Optimizer

from ..api import Resolver

__all__ = [
    "optimizer_resolver",
    "activation_resolver",
]

optimizer_resolver = Resolver.from_subclasses(Optimizer, default=Adam, suffix="")

activation_resolver = Resolver(
    classes=(
        nn.LeakyReLU,
        nn.PReLU,
        nn.ReLU,
        nn.Softplus,
        nn.Sigmoid,
        nn.Tanh,
    ),
    base=nn.Module,
    suffix="",
    default=nn.ReLU,
)

if __name__ == '__main__':
    print(activation_resolver.options)
    print(optimizer_resolver.options)
