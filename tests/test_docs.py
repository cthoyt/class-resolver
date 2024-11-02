"""Tests for documenting resolvers."""

from __future__ import annotations

import unittest
from typing import Any

from torch import Tensor, nn

from class_resolver import FunctionResolver, ResolverKey, update_docstring_with_resolver_keys
from class_resolver.contrib.torch import activation_resolver, aggregation_resolver
from class_resolver.docs import _clean_docstring

TARGET = """This method does some stuff

:param a: Something about A
:param b: Something about B
""".rstrip()

DS1 = """This method does some stuff

:param a: Something about A
:param b: Something about B
"""

DS2 = """This method does some stuff

    :param a: Something about A
    :param b: Something about B
"""

DS3 = """This method does some stuff

        :param a: Something about A
        :param b: Something about B
"""

DS4 = """\
This method does some stuff

:param a: Something about A
:param b: Something about B
"""

DS5 = """\
    This method does some stuff

    :param a: Something about A
    :param b: Something about B
"""


TEST_RESOLVER_1 = update_docstring_with_resolver_keys(
    ResolverKey("activation", "class_resolver.contrib.torch.activation_resolver"),
)

TEST_RESOLVER_2 = update_docstring_with_resolver_keys(
    ResolverKey("activation", activation_resolver),
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


@update_docstring_with_resolver_keys(
    ResolverKey("activation", "class_resolver.contrib.torch.activation_resolver"),
    ResolverKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
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


@update_docstring_with_resolver_keys(
    ResolverKey("activation_1", "class_resolver.contrib.torch.activation_resolver"),
    ResolverKey("activation_2", "class_resolver.contrib.torch.activation_resolver"),
    ResolverKey("aggregation", "class_resolver.contrib.torch.aggregation_resolver"),
)
def f4(
    tensor: Tensor,
    activation_1: None | str | type[nn.Module] | nn.Module,
    activation_1_kwargs: dict[str, Any] | None,
    aggregation: None | str | type[nn.Module] | nn.Module,
    aggregation_kwargs: dict[str, Any] | None,
    activation_2: None | str | type[nn.Module] | nn.Module,
    activation_2_kwargs: dict[str, Any] | None,
):
    """Apply an activation then aggregation.

    :param tensor: An input tensor
    :param activation_1: An activation function (stateful)
    :param activation_1_kwargs: Keyword arguments for activation function
    :param aggregation: An aggregation function (stateful)
    :param aggregation_kwargs: Keyword arguments for aggregation function
    :param activation_2: An activation function (stateful)
    :param activation_2_kwargs: Keyword arguments for activation function

    :return: An aggregation
    """
    _activation_1 = activation_resolver.make(activation_1, activation_1_kwargs)
    _activation_2 = activation_resolver.make(activation_2, activation_2_kwargs)
    _aggregation = aggregation_resolver.make(aggregation, aggregation_kwargs)
    return _activation_2(_aggregation(_activation_2(tensor)))


EXPECTED_FUNCTION_4_DOC = """\
Apply an activation then aggregation.

:param tensor: An input tensor
:param activation_1: An activation function (stateful)
:param activation_1_kwargs: Keyword arguments for activation function
:param aggregation: An aggregation function (stateful)
:param aggregation_kwargs: Keyword arguments for aggregation function
:param activation_2: An activation function (stateful)
:param activation_2_kwargs: Keyword arguments for activation function

:return: An aggregation

.. note ::

    3 resolvers are used in this function.

    - The parameter pairs ``(activation_1, activation_1_kwargs)``, ``(activation_2, activation_2_kwargs)`` are used for :data:`class_resolver.contrib.torch.activation_resolver`
    - The parameter pair ``(aggregation, aggregation_kwargs)`` is used for :data:`class_resolver.contrib.torch.aggregation_resolver`

    An explanation of resolvers and how to use them is given in
    https://class-resolver.readthedocs.io/en/latest/.
""".rstrip()


@TEST_RESOLVER_2
def f5(activation, activation_kwargs):
    """Apply an activation then aggregation.

    :param activation: An activation function (stateful)
    :param activation_kwargs: Keyword arguments for activation function
    """


class DecoratorTests(unittest.TestCase):
    """Decorator tests."""

    @staticmethod
    def f(model, model_kwargs) -> None:
        """Do something, and also use model."""
        pass

    # docstr-coverage:excused `testing missing docstr on purpose`
    @staticmethod
    def f_no_doc(model, model_kwargs) -> None:  # noqa: D102
        pass

    def test_decorator(self):
        """Test decorator."""
        old_doc = self.f.__doc__
        for params in [("model", "model_resolver"), ("model", "model_resolver", "model_kwargs")]:
            with self.subTest(params=params):
                decorator = update_docstring_with_resolver_keys(ResolverKey(*params))
                f_dec = decorator(self.f)
                # note: the decorator modifies the doc string in-place...
                # check that the doc string got extended
                self.assertNotEqual(f_dec.__doc__, old_doc)
                self.assertTrue(f_dec.__doc__.startswith(old_doc))
                # revert for next time
                self.f.__doc__ = old_doc

    def test_error_decoration(self):
        """Test errors when decorating."""
        # missing docstring
        with self.assertRaises(ValueError):
            update_docstring_with_resolver_keys(ResolverKey("model", "model_resolver"))(self.f_no_doc)
        # non-existing parameter name
        with self.assertRaises(ValueError):
            update_docstring_with_resolver_keys(ResolverKey("interaction", "model_resolver"))(self.f)


class TestDocumentResolver(unittest.TestCase):
    """Test documenting resolvers."""

    def test_clean_docstring(self) -> None:
        """Test cleaning a docstring works correctly."""
        for ds in [TARGET, DS1, DS2, DS3, DS4, DS5]:
            with self.subTest(docstring=ds):
                self.assertEqual(TARGET, _clean_docstring(ds))

    def test_bad_type(self):
        """Raise the appropriate error."""
        with self.assertRaises(TypeError):
            ResolverKey("", None)

    def test_no_params(self):
        """Test when no keys are passed."""
        with self.assertRaises(ValueError):
            update_docstring_with_resolver_keys()

    def test_duplicate_params(self):
        """Test when no keys are passed."""
        key = ResolverKey("a", "b")
        with self.assertRaises(ValueError):
            update_docstring_with_resolver_keys(key, key)

    def test_missing_params(self):
        """Test when trying to document a parameter that does not exist."""
        with self.assertRaises(ValueError):

            @update_docstring_with_resolver_keys(ResolverKey("some-other-param", "y"))
            def f(x):
                """Do the thing."""

    def test_no_location(self):
        """Test when there's no explicit location given."""
        r = FunctionResolver([])
        with self.assertRaises(NotImplementedError):
            ResolverKey("xx", r)

    def test_f1(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_1_DOC, f1.__doc__)

    def test_f2(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_1_DOC, f2.__doc__)

    def test_f3(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_3_DOC, f3.__doc__)

    def test_f4(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_4_DOC, f4.__doc__)

    def test_f5(self):
        """Test the correct docstring is produced."""
        self.assertEqual(EXPECTED_FUNCTION_1_DOC, f5.__doc__)


class TestTable(unittest.TestCase):
    """Test building tables for resolvers."""

    def test_activation_resolver(self):
        """Test activation resolver table."""
        tab = activation_resolver.make_table()
        lines = tab.splitlines()
        # these are the separator "=== ===" lines
        for index in (0, 2, -1):
            self.assertEqual(set(lines[index]), {" ", "="})
        # the header
        self.assertEqual({x.strip() for x in lines[1].split()}, {"key", "class"})
        # check content rows
        for line in lines[3:-1]:
            parts = line.split()
            self.assertEqual(len(parts), 2)
            key, cls = parts
            cls = cls.strip()
            self.assertTrue(cls.startswith(":class:`~"))
            cls = cls.removeprefix(":class:`~").removesuffix("`")
            key = key.strip("`")
            o_cls = activation_resolver.lookup(key.strip())
            self.assertEqual(f"{o_cls.__module__}.{o_cls.__qualname__}", cls)
