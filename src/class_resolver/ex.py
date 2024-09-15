"""Example.

Delete this before merging
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from class_resolver import document_resolver
from class_resolver.contrib.torch import activation_resolver, aggregation_resolver

__all__ = ["f"]


@document_resolver("activation", resolver_name="class_resolver.contrib.torch.activation_resolver")
@document_resolver("aggregation", resolver_name="class_resolver.contrib.torch.aggregation_resolver")
def f(
    tensor: Tensor,
    activation: None | str | type[nn.Module] | nn.Module,
    activation_kwargs: dict[str, Any] | None,
    aggregation: None | str | type[nn.Module] | nn.Module,
    aggregation_kwargs: dict[str, Any] | None,
):
    """Apply an activation then aggregation.

    :param tensor: An input tensor
    :param activation: An activation function (stateful)
    :param activation_kwargs: Keyword arguments for activation function
    :param aggregation: An aggregation function (stateful)
    :param aggregation_kwargs: Keyword arguments for aggregation function

    :return: An aggregation
    """
    _activation = activation_resolver.make(activation, activation_kwargs)
    _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
    return _aggregation(_activation(tensor))
