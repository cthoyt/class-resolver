import unittest
from typing import Optional, Type

from class_resolver import ClassResolver, Hint, OptionalKwargs
from class_resolver.metaresolver import Metaresolver, is_hint


class Baz:
    def __init__(self, value: bool = False):
        self.value = value


class XBaz(Baz):
    pass


class YBaz(Baz):
    pass


baz_resolver = ClassResolver.from_subclasses(Baz)


class Bar:
    def __init__(
        self,
        baz: Hint[Baz] = None,
        baz_kwargs: OptionalKwargs = None,
    ):
        self.baz = baz_resolver.make(baz, baz_kwargs)


class AlphaBar(Bar):
    pass


class BetaBar(Bar):
    pass


bar_resolver = ClassResolver.from_subclasses(Bar)


class Foo:
    def __init__(
        self,
        *,
        bar: Hint[Bar] = None,
        bar_kwargs: OptionalKwargs = None,
        param_1: float,
        param_2: Optional[int] = None,
    ):
        self.bar = bar_resolver.make(bar, bar_kwargs)
        self.param_1 = param_1
        self.param_2 = param_2 or 5


class AFoo(Foo):
    pass


class BFoo(Foo):
    pass


foo_resolver = ClassResolver.from_subclasses(Foo)


class TestMetaResolver(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.meta_resolver = Metaresolver([baz_resolver, bar_resolver, foo_resolver])

    def test_is_hint(self):
        """Test hint predicate."""
        self.assertTrue(is_hint(Hint[Foo], Foo))
        self.assertFalse(is_hint(Hint[Foo], Bar))
        self.assertFalse(is_hint(Type[Bar], Bar))
        self.assertFalse(is_hint(str, Bar))
        self.assertFalse(is_hint(None, Bar))

    def test_check(self):
        self.assertTrue(self.meta_resolver.check_kwargs(
            Foo, AFoo,
            {
                "bar": "alpha",
                "bar_kwargs": {
                    "baz": "x",
                    "baz_kwargs": {
                        "value": True,
                    }
                },
                "param_1": 3.0,
                # Param 2 is optional, so not necessary to give
            },
        ))
        self.assertTrue(self.meta_resolver.check_kwargs(
            Foo, AFoo,
            {
                "bar": "alpha",
                "bar_kwargs": {
                    "baz": "x",
                    # baz_kwargs value not necessary since has default
                },
                "param_1": 3.0,
                # Param 2 is optional, so not necessary to give
            },
        ))
        self.assertFalse(self.meta_resolver.check_kwargs(
            Foo, AFoo,
            {
                "bar": "alpha",
                "bar_kwargs": {
                    "baz": "x",
                },
                # Missing param_1 !!
            },
        ))
