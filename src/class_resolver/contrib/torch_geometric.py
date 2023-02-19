# -*- coding: utf-8 -*-

"""
PyTorch Geometric is an extension to PyTorch for geometric learning on graphs,
point clouds, meshes, and other non-standard objects.
The ``class-resolver`` provides several class resolvers and function resolvers
to make it possible to more easily parametrize models and training loops.
"""  # noqa:D205,D400

from torch_geometric.nn.aggr import Aggregation, MeanAggregation
from torch_geometric.nn.conv import MessagePassing, SimpleConv

from ..api import ClassResolver

__all__ = [
    "message_passing_resolver",
    "aggregation_resolver",
]

message_passing_resolver = ClassResolver.from_subclasses(
    base=MessagePassing,  # type: ignore
    suffix="Conv",
    default=SimpleConv,
)
"""A resolver for message passing layers.

.. seealso:: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers
"""

aggregation_resolver = ClassResolver.from_subclasses(
    base=Aggregation,
    default=MeanAggregation,
)

"""A resolver for aggregation layers.

.. seealso:: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
"""
