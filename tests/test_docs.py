"""Tests for documenting resolvers."""

from __future__ import annotations

import unittest
from textwrap import dedent
from typing import Any

from torch import Tensor, nn

from class_resolver import DocKey, document_resolver
from class_resolver.contrib.torch import activation_resolver, aggregation_resolver


@document_resolver(
    DocKey("activation", "class_resolver.contrib.torch.activation_resolver"),
    DocKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
)
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


EXPECTED = """\
Apply an activation then aggregation.

:param tensor: An input tensor
:param activation: An activation function (stateful)
:param activation_kwargs: Keyword arguments for activation function
:param aggregation: An aggregation function (stateful)
:param aggregation_kwargs: Keyword arguments for aggregation function

:return: An aggregation

.. note ::

    2 resolvers are used in this function.

    - The parameter pairs ``(activation, activation_kwargs)`` are used for :data:`class_resolver.contrib.torch.activation_resolver`
    - The parameter pairs ``(aggregation, aggregation_kwargs)`` are used for :data:`class_resolver.contrib.torch.aggregation_resolver`

    An explanation of resolvers and how to use them is given in
    https://class-resolver.readthedocs.io/en/latest/.

"""


class TestDocumentResolver(unittest.TestCase):
    """Test documenting resolvers."""

    def test_f(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED, dedent(f.__doc__))