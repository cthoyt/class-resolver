# -*- coding: utf-8 -*-

"""Class resolvers for PyTorch."""

from torch import nn
from torch.nn import init
from torch.nn.modules import activation
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

ACTIVATION_SKIP = {
    nn.MultiheadAttention,
    nn.Softmax2d,
}
activation_resolver = Resolver(
    classes=[
        module
        for module in vars(activation).values()
        if isinstance(module, type)
        and issubclass(module, nn.Module)
        and module not in ACTIVATION_SKIP
    ],
    base=nn.Module,
    default=nn.ReLU,
    base_as_suffix=False,
)

initializer_resolver = FunctionResolver(
    [func for name, func in vars(init).items() if not name.startswith("_") and name.endswith("_")],
    default=init.normal_,
)
