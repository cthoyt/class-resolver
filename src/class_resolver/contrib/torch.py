# -*- coding: utf-8 -*-

"""
PyTorch is a tensor and autograd library widely used for machine learning.
The ``class-resolver`` provides several class resolvers and function resolvers
to make it possible to more easily parametrize models and training loops.
"""  # noqa:D205,D400

from torch import nn
from torch.nn import init
from torch.nn.modules import activation
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau, _LRScheduler

from ..api import ClassResolver
from ..func import FunctionResolver

__all__ = [
    "optimizer_resolver",
    "activation_resolver",
    "margin_activation_resolver",
    "initializer_resolver",
    "lr_scheduler_resolver",
]

optimizer_resolver = ClassResolver.from_subclasses(
    Optimizer,
    default=Adam,
    base_as_suffix=False,
)
"""A resolver for :class:`torch.optim.Optimizer` classes.

.. code-block:: python

    from class_resolver import Hint, OptionalKwargs
    from class_resolver.contrib.torch import optimizer_resolver
    from torch import Parameter
    from torch.optim import Optimizer

    def train(optimizer: Hint[Optimizer] = "adam", optimizer_kwargs: OptionalKwargs = None):
        model = [Parameter(torch.randn(2, 2, requires_grad=True))]
        optimizer = optimizer_resolver.make(optimizer, optimizer_kwargs, model=model)

        for epoch in range(20):
            for input, target in dataset:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()

        return model

"""

ACTIVATION_SKIP = {
    activation.MultiheadAttention,
    activation.Softmax2d,
}

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
"""A resolver for :mod:`torch.nn.modules.activation` classes.

.. code-block:: python

    import torch
    from class_resolver.contrib.torch import activation_resolver
    from more_itertools import pairwise
    from torch import nn
    from torch.nn import functional as F

    class TwoLayerPerceptron(nn.Module):
        def __init__(
            self,
            dims: list[int]
            activation: Hint[nn.Module] = None
        )
            layers = []
            for in_features, out_features in pairwise(dims):
                layers.extend((
                    nn.Linear(in_features, out_features),
                    activation_resolver.make(activation),
                ))
            self.layers = nn.Sequential(*layers)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            return self.layers(x)
"""

margin_activation_resolver = ClassResolver(
    classes={
        nn.ReLU,
        nn.Softplus,
    },
    base=nn.Module,  # type: ignore
    default=nn.ReLU,
    synonyms=dict(
        hard=nn.ReLU,
        soft=nn.Softplus,
    ),
)
"""A resolver for a subset of :mod:`torch.nn.modules.activation` classes.

This resolver fulfills the same idea as :data:`activation_resolver` but
it is explicitly limited to :class:`torch.nn.ReLU` and :class:`torch.nn.Softplus`
for certain scenarios where a margin-style activation is appropriate.
"""

initializer_resolver = FunctionResolver(
    [func for name, func in vars(init).items() if not name.startswith("_") and name.endswith("_")],
    default=init.normal_,
)
"""A resolver for :mod:`torch.nn.init` functions.

.. code-block:: python

    import torch
    from class_resolver.contrib.torch import initializer_resolver
    from torch import nn
    from torch.nn import functional as F

    class TwoLayerPerceptron(nn.Module):
        def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            initializer=nn.init.xavier_normal_,
        )
            self.layer_1 = nn.Linear(in_features, hidden_features)
            self.layer_2 = nn.Linear(hidden_features, out_features)

            initializer = initializer_resolver.lookup(initializer)
            initializer(self.layer_1.weights)
            initializer(self.layer_1.weights)

        def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
            x = self.layer_1(x)
            x = F.relu(x)
            x = self.layer_2(x)
            x = F.relu(x)
            return x
"""

lr_scheduler_resolver = ClassResolver.from_subclasses(
    _LRScheduler,
    default=ExponentialLR,
    suffix="LR",
)
"""A resolver for learning rate schedulers.

Borrowing from the PyTorch documentation's example on `how to adjust the learning
rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_,
the following example shows how a training loop can be first turned into a funciton
then parametrized to accept a LRScheduler hint.

.. code-block:: python

    from class_resolver import Hint, OptionalKwargs
    from class_resolver.contrib.torch import lr_scheduler_resolver
    from torch import Parameter
    from torch.optim import SGD
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    def train(scheduler: Hint[LRScheduler] = "exponential", scheduler_kwargs: OptionalKwargs = None):
        model = [Parameter(torch.randn(2, 2, requires_grad=True))]
        optimizer = SGD(model, 0.1)
        scheduler = lr_scheduler_resolver.make(scheduler, scheduler_kwargs, optimizer=optimizer)

        for epoch in range(20):
            for input, target in dataset:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
            scheduler.step()

        return model
"""

lr_scheduler_resolver.register(ReduceLROnPlateau)
