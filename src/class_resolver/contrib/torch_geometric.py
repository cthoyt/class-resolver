"""
PyTorch Geometric is an extension to PyTorch for geometric learning on graphs,
point clouds, meshes, and other non-standard objects.
The ``class-resolver`` provides several class resolvers and function resolvers
to make it possible to more easily parametrize models and training loops.
"""  # noqa: D205

from torch_geometric.nn.aggr import Aggregation, MeanAggregation
from torch_geometric.nn.conv import MessagePassing, SimpleConv

from ..api import ClassResolver

__all__ = [
    "aggregation_resolver",
    "message_passing_resolver",
]

message_passing_resolver = ClassResolver.from_subclasses(
    base=MessagePassing,
    suffix="Conv",
    default=SimpleConv,
    location="class_resolver.contrib.torch_geometric.message_passing_resolver",
)
"""A resolver for message passing layers.

.. seealso:: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#convolutional-layers
"""

aggregation_resolver = ClassResolver.from_subclasses(
    base=Aggregation,
    default=MeanAggregation,
    location="class_resolver.contrib.torch_geometric.aggregation_resolver",
)
"""A resolver for aggregation layers.

This includes the following:

- :class:`torch_geometric.nn.aggr.MeanAggregation`
- :class:`torch_geometric.nn.aggr.MaxAggregation`
- :class:`torch_geometric.nn.aggr.MinAggregation`
- :class:`torch_geometric.nn.aggr.SumAggregation`
- :class:`torch_geometric.nn.aggr.MedianAggregation`
- :class:`torch_geometric.nn.aggr.SoftmaxAggregation` (learnable)
- :class:`torch_geometric.nn.aggr.PowerMeanAggregation` (learnable)
- :class:`torch_geometric.nn.aggr.LSTMAggregation` (learnable)
- :class:`torch_geometric.nn.aggr.MLPAggregation` (learnable)
- :class:`torch_geometric.nn.aggr.SetTransformerAggregation)` (learnable)
- :class:`torch_geometric.nn.aggr.SortAggregation`

Some example usage (based on the torch-geometric docs):

.. code-block::

    import torch

    from class_resolver.contrib.torch_geometric import aggregation_resolver

    mean_aggr = aggregation_resolver.make("mean")

    # Feature matrix holding 1000 elements with 64 features each:
    x = torch.randn(1000, 64)

    # Randomly assign elements to 100 sets:
    index = torch.randint(0, 100, (1000, ))

    output = mean_aggr(x, index)  #  Output shape: [100, 64]

.. seealso:: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#aggregation-operators
"""
