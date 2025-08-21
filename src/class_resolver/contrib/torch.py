"""
PyTorch is a tensor and autograd library widely used for machine learning.
The ``class-resolver`` provides several class resolvers and function resolvers
to make it possible to more easily parametrize models and training loops.
"""  # noqa: D205

from collections.abc import Callable
from typing import TypeAlias

import torch
from torch import nn
from torch.nn import init
from torch.nn.modules import activation
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau

try:
    # when torch >= 2.0
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # when torch < 2.0
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from ..api import ClassResolver
from ..func import FunctionResolver

__all__ = [
    "activation_resolver",
    "aggregation_resolver",
    "initializer_resolver",
    "lr_scheduler_resolver",
    "margin_activation_resolver",
    "optimizer_resolver",
]

optimizer_resolver = ClassResolver.from_subclasses(
    Optimizer,
    default=Adam,
    base_as_suffix=False,
    location="class_resolver.contrib.torch.optimizer_resolver",
)
"""A resolver for :class:`torch.optim.Optimizer` classes.

.. code-block:: python

    from class_resolver import Hint, OptionalKwargs
    from class_resolver.contrib.torch import optimizer_resolver
    from torch import Parameter, nn
    from torch.optim import Optimizer

    dataset = ...

    def train(
        model: nn.Module,
        optimizer: Hint[Optimizer] = "adam",
        optimizer_kwargs: OptionalKwargs = None,
    ):
        optimizer = optimizer_resolver.make(
            optimizer, optimizer_kwargs, params=model.parameters(),
        )

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

activation_resolver: ClassResolver[nn.Module] = ClassResolver(
    classes=[
        module
        for module in vars(activation).values()
        if isinstance(module, type) and issubclass(module, nn.Module) and module not in ACTIVATION_SKIP
    ],
    base=nn.Module,
    default=activation.ReLU,
    base_as_suffix=False,
    location="class_resolver.contrib.torch.activation_resolver",
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

margin_activation_resolver: ClassResolver[nn.Module] = ClassResolver(
    classes={
        nn.ReLU,
        nn.Softplus,
    },
    base=nn.Module,
    default=nn.ReLU,
    synonyms={
        "hard": nn.ReLU,
        "soft": nn.Softplus,
    },
    location="class_resolver.contrib.torch.margin_activation_resolver",
)
"""A resolver for a subset of :mod:`torch.nn.modules.activation` classes.

This resolver fulfills the same idea as :data:`activation_resolver` but
it is explicitly limited to :class:`torch.nn.ReLU` and :class:`torch.nn.Softplus`
for certain scenarios where a margin-style activation is appropriate.
"""

initializer_resolver = FunctionResolver(
    [func for name, func in vars(init).items() if not name.startswith("_") and name.endswith("_")],
    default=init.normal_,
    location="class_resolver.contrib.torch.initializer_resolver",
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
    LRScheduler,
    default=ExponentialLR,
    suffix="LR",
    location="class_resolver.contrib.torch.lr_scheduler_resolver",
)
"""A resolver for learning rate schedulers.

Borrowing from the PyTorch documentation's example on `how to adjust the learning
rate <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_,
the following example shows how a training loop can be first turned into a funciton
then parametrized to accept a LRScheduler hint.

.. code-block:: python

    from class_resolver import Hint, OptionalKwargs
    from class_resolver.contrib.torch import lr_scheduler_resolver
    from torch import Parameter, nn
    from torch.optim import SGD
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

    dataset = ...

    def train(
        model: nn.Module,
        scheduler: Hint[LRScheduler] = "exponential",
        scheduler_kwargs: OptionalKwargs = None,
    ):
        optimizer = SGD(params=model.parameters(), lr=0.1)
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
# this is for torch<2.1 compat
lr_scheduler_resolver.register(ReduceLROnPlateau, raise_on_conflict=False)

TorchAggregationFunc: TypeAlias = Callable[..., torch.Tensor]

_AGGREGATION_FUNCTIONS: list[TorchAggregationFunc] = [
    torch.sum,
    torch.max,
    torch.min,
    torch.mean,
    torch.logsumexp,
    torch.median,
]

aggregation_resolver = FunctionResolver(
    _AGGREGATION_FUNCTIONS,
    default=torch.mean,
    location="class_resolver.contrib.torch.aggregation_resolver",
)
"""A resolver for common aggregation functions in PyTorch including the following functions:

- :func:`torch.sum`
- :func:`torch.max`
- :func:`torch.min`
- :func:`torch.mean`
- :func:`torch.median`
- :func:`torch.logsumexp`

The default value is :func:`torch.mean`. This resolver can be used like in the
following:

.. code-block:: python

    import torch
    from class_resolver.contrib.torch import aggregation_resolver

    # Lookup with string
    func = aggregation_resolver.lookup("max")
    arr = torch.tensor([1, 2, 3, 10], dtype=torch.float)
    assert 10.0 == func(arr).item()

    # Default lookup gives mean
    func = aggregation_resolver.lookup(None)
    arr = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float)
    assert 4.0 == func(arr).item()

    def first(x):
        return x[0]

    # Custom functions pass through
    func = aggregation_resolver.lookup(first)
    arr = torch.tensor([1.0, 2.0, 3.0, 10.0], dtype=torch.float)
    assert 1.0 == func(arr).item()
"""
