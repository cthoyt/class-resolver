"""NumPy is a numerical package for Python."""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from ..func import FunctionResolver

__all__ = [
    "aggregation_resolver",
]

NumpyAggregationFunc: TypeAlias = Callable[..., ArrayLike]

_AGGREGATION_FUNCTIONS: list[NumpyAggregationFunc] = [np.sum, np.max, np.min, np.mean, np.median]

aggregation_resolver = FunctionResolver(
    _AGGREGATION_FUNCTIONS,
    default=np.mean,
    location="class_resolver.contrib.numpy.aggregation_resolver",
)
"""A resolver for common aggregation functions in NumPy including the following functions:

- :func:`numpy.sum`
- :func:`numpy.max`
- :func:`numpy.min`
- :func:`numpy.mean`
- :func:`numpy.median`

The default value is :func:`numpy.mean`. This resolver can be used like in the
following:

.. code-block:: python

    from class_resolver.contrib.numpy import aggregation_resolver

    # Lookup with string
    func = aggregation_resolver.lookup("max")
    arr = [1, 2, 3, 10]
    assert 10 == func(arr).item()

    # Default lookup gives mean
    func = aggregation_resolver.lookup(None)
    arr = [1, 2, 3, 10]
    assert 4.0 == func(arr).item()

    def first(x):
        return x[0]

    # Custom functions pass through
    func = aggregation_resolver.lookup(first)
    arr = [1, 2, 3, 10]
    assert 1 == func(arr)
"""

# compat with older numpy versions, where np.min points to np.amin
aggregation_resolver.register(np.min, synonyms={"min"}, raise_on_conflict=False)
aggregation_resolver.register(np.max, synonyms={"max"}, raise_on_conflict=False)
