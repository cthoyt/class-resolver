# -*- coding: utf-8 -*-

"""Class resolvers for PyTorch."""

from torch import nn
from torch.optim import Adam, Optimizer

from ..api import Resolver
from ..func import FunctionResolver

__all__ = [
    "optimizer_resolver",
    "activation_resolver",
    "initializer_resolver",
]

optimizer_resolver = Resolver.from_subclasses(
    Optimizer,
    default=Adam,
    base_as_suffix=False,
)

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
    default=nn.ReLU,
    base_as_suffix=False,
)

initializer_resolver = FunctionResolver(
    [
        getattr(nn.init, func)
        for func in dir(nn.init)
        if not func.startswith("_") and func.endswith("_")
    ],
    default=nn.init.normal_,
)

if __name__ == "__main__":
    print(activation_resolver.options)
    print(optimizer_resolver.options)
    print(initializer_resolver.options)
