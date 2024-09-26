"""Tests for documenting resolvers."""

from __future__ import annotations

import unittest
from typing import Any

from torch import Tensor, nn

from class_resolver import DocKey, document_resolver
from class_resolver.contrib.torch import activation_resolver, aggregation_resolver

TEST_RESOLVER_1 = document_resolver(
    DocKey("activation", "class_resolver.contrib.torch.activation_resolver"),
)

EXPECTED_FUNCTION_1_DOC = """\
Apply an activation then aggregation.

:param activation: An activation function (stateful)
:param activation_kwargs: Keyword arguments for activation function

.. note ::

    The parameter pair ``(activation, activation_kwargs)`` is used for :data:`class_resolver.contrib.torch.activation_resolver`

    An explanation of resolvers and how to use them is given in
    https://class-resolver.readthedocs.io/en/latest/.
""".rstrip()


@TEST_RESOLVER_1
def f1(activation, activation_kwargs):
    """Apply an activation then aggregation.

    :param activation: An activation function (stateful)
    :param activation_kwargs: Keyword arguments for activation function
    """


@TEST_RESOLVER_1
def f2(activation, activation_kwargs):
    """Apply an activation then aggregation.

    :param activation: An activation function (stateful)
    :param activation_kwargs: Keyword arguments for activation function
    """


@document_resolver(
    DocKey("activation", "class_resolver.contrib.torch.activation_resolver"),
    DocKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
)
def f3(
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


EXPECTED_FUNCTION_3_DOC = """\
Apply an activation then aggregation.

:param tensor: An input tensor
:param activation: An activation function (stateful)
:param activation_kwargs: Keyword arguments for activation function
:param aggregation: An aggregation function (stateful)
:param aggregation_kwargs: Keyword arguments for aggregation function

:return: An aggregation

.. note ::

    2 resolvers are used in this function.

    - The parameter pair ``(activation, activation_kwargs)`` is used for :data:`class_resolver.contrib.torch.activation_resolver`
    - The parameter pair ``(aggregation, aggregation_kwargs)`` is used for :data:`class_resolver.contrib.torch.aggregation_resolver`

    An explanation of resolvers and how to use them is given in
    https://class-resolver.readthedocs.io/en/latest/.
""".rstrip()


class TestDocumentResolver(unittest.TestCase):
    """Test documenting resolvers."""

    def test_f1(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_1_DOC, f1.__doc__)

    def test_f2(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_1_DOC, f2.__doc__)

    def test_f3(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_3_DOC, f3.__doc__)
