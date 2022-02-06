# -*- coding: utf-8 -*-

"""Class resolvers for PyTorch."""

from torch import nn
from torch.nn import init
from torch.nn.modules import activation
from torch.optim import Adam, Optimizer

from ..api import ClassResolver
from ..func import FunctionResolver

__all__ = [
    "optimizer_resolver",
    "activation_resolver",
    "initializer_resolver",
]

#: A resolver for :class:`torch.optim.Optimizer` classes
optimizer_resolver = ClassResolver.from_subclasses(
    Optimizer,
    default=Adam,
    base_as_suffix=False,
)

ACTIVATION_SKIP = {
    activation.MultiheadAttention,
    activation.Softmax2d,
}

#: A resolver for :mod:`torch.nn.modules.activation` classes
activation_resolver = ClassResolver(
    classes=[
        module
        for module in vars(activation).values()
        if isinstance(module, type)
        and issubclass(module, nn.Module)
        and module not in ACTIVATION_SKIP
    ],
    base=nn.Module,
    default=activation.ReLU,
    base_as_suffix=False,
)

#: A resolver for :mod:`torch.nn.init` functions
initializer_resolver = FunctionResolver(
    [func for name, func in vars(init).items() if not name.startswith("_") and name.endswith("_")],
    default=init.normal_,
)
