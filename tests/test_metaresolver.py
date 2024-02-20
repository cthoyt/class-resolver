# -*- coding: utf-8 -*-

"""Tests for the argument checker."""

import typing as t
import unittest
from typing import Optional, Type

from class_resolver import ClassResolver, Hint, OptionalKwargs
from class_resolver.metaresolver import ArgumentError, check_kwargs, is_hint


class Baz:
    """A dummy class."""

    def __init__(self, value: bool = False):
        """Instantiate the dummy class."""
        self.value = value


class XBaz(Baz):
    """A dummy child of the Baz class."""


class YBaz(Baz):
    """A dummy child of the Baz class."""


baz_resolver = ClassResolver.from_subclasses(Baz)


class Bar:
    """A dummy class."""

    def __init__(
        self,
        baz: Hint[Baz] = None,
        baz_kwargs: OptionalKwargs = None,
    ):
        """Instantiate the dummy class."""
        self.baz = baz_resolver.make(baz, baz_kwargs)


class AlphaBar(Bar):
    """A dummy child of the Bar class."""


class BetaBar(Bar):
    """A dummy child of the Bar class."""


bar_resolver = ClassResolver.from_subclasses(Bar)


class Foo:
    """A dummy class."""

    def __init__(
        self,
        *,
        bar: Hint[Bar] = None,
        bar_kwargs: OptionalKwargs = None,
        param_1: float,
        param_2: Optional[int] = None,
    ):
        """Instantiate the dummy class."""
        self.bar = bar_resolver.make(bar, bar_kwargs)
        self.param_1 = param_1
        self.param_2 = param_2 or 5


class AFoo(Foo):
    """A dummy child of the Foo class."""


class BFoo(Foo):
    """A dummy child of the Foo class."""


foo_resolver = ClassResolver.from_subclasses(Foo)


class TestMetaResolver(unittest.TestCase):
    """A test case for the argument checker."""

    def setUp(self) -> None:
        """Set up the test case."""
        self.resolvers = [baz_resolver, bar_resolver, foo_resolver]

    def check_kwargs(self, func, kwargs) -> bool:
        """Check the kwargs."""
        return check_kwargs(func, kwargs, resolvers=self.resolvers)

    def test_is_hint(self):
        """Test hint predicate."""
        self.assertTrue(is_hint(Hint[Foo], Foo))
        self.assertFalse(is_hint(Hint[Foo], Bar))
        self.assertFalse(is_hint(Type[Bar], Bar))
        self.assertFalse(is_hint(str, Bar))
        self.assertFalse(is_hint(None, Bar))
        with self.assertRaises(TypeError):
            is_hint(..., 5)

    def test_annotation_resolver_mismatch(self):
        """Test when the key for a resolver is annotated incorrectly."""

        class Bax:
            """A dummy class."""

            def __init__(self, baz: int):
                """Instantiate the dummy class."""

        with self.assertRaises(ArgumentError):
            self.check_kwargs(Bax, {"baz": "x"})

    def test_annotation_mismatch(self):
        """Test when a random value is annotated incorrectly."""

        class Bax:
            """A dummy class."""

            def __init__(self, value: Optional[int]):
                """Instantiate the dummy class."""
                self.value = value

        self.assertTrue(self.check_kwargs(Bax, {"value": 5}))

    def test_annotation_extended_origin_mismatch(self):
        """Test when a random value is annotated incorrectly."""

        class Bax:
            """A dummy class."""

            def __init__(self, value: t.Union[int, float, None]):
                """Instantiate the dummy class."""

        with self.assertRaises(ArgumentError):
            self.check_kwargs(Bax, {"value": "nope"})

    def test_annotation_extended_origin_mismatch_2(self):
        """Test when a random value is annotated incorrectly."""

        class Bax:
            """A dummy class."""

            def __init__(self, value: t.Union[int, float, None, t.Iterable[int]]):
                """Instantiate the dummy class."""

        with self.assertRaises(ArgumentError):
            self.check_kwargs(Bax, {"value": 1})

    def test_annotation_invalid_origin(self):
        """Test when a random value is annotated incorrectly."""

        class Bax:
            """A dummy class."""

            def __init__(self, value: t.Iterable[int]):
                """Instantiate the dummy class."""

        with self.assertRaises(ArgumentError):
            self.check_kwargs(Bax, {"value": 5})

    def test_argument_checker(self):
        """Test the argument checker."""
        true_kwargs = [
            (
                AFoo,
                {
                    "bar": "alpha",
                    "bar_kwargs": {
                        "baz": "x",
                        "baz_kwargs": {
                            "value": True,
                        },
                    },
                    "param_1": 3.0,
                    # Param 2 is optional, so not necessary to give
                },
            ),
            (
                AFoo,
                {
                    "bar": "alpha",
                    "bar_kwargs": {
                        "baz": "x",
                        "baz_kwargs": {
                            "value": True,
                        },
                    },
                    "param_1": 3.0,
                    "param_2": 2,
                },
            ),
        ]
        for func, kwargs in true_kwargs:
            with self.subTest():
                self.assertTrue(self.check_kwargs(func, kwargs))

        false_kwargs = [
            (
                AFoo,
                {
                    "bar": "alpha",
                    "bar_kwargs": {
                        "baz": "x",
                    },
                    # Missing param_1 !!
                },
            ),
            (
                AFoo,
                {
                    "bar": "alpha",
                    "bar_kwargs": {
                        "baz": "x",
                    },
                    "param_1": "3.0",  # wrong type, should be float
                },
            ),
            (AFoo, None),
            (AFoo, {}),
        ]
        for func, kwargs in false_kwargs:
            with self.subTest(), self.assertRaises(ArgumentError):
                self.check_kwargs(func, kwargs)
