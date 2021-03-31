# -*- coding: utf-8 -*-

"""Tests for the class resolver."""
import itertools
import unittest

from class_resolver import Resolver

try:
    import ray.tune as tune
except ImportError:
    tune = None


class Base:
    """A base class."""

    def __init__(self, name):
        """Initialize the class."""
        self.name = name

    def __eq__(self, other) -> bool:
        """Check two instances are equal."""
        return type(self) == type(other) and self.name == other.name


class A(Base):
    """A base class."""


class B(Base):
    """B base class."""


class C(Base):
    """C base class."""


class TestResolver(unittest.TestCase):
    """Tests for the resolver."""

    def setUp(self) -> None:
        """Set up the resolver class."""
        self.resolver = Resolver([A, B, C], base=Base)

    def test_lookup(self):
        """Test looking up classes."""
        self.assertEqual(A, self.resolver.lookup('a'))
        self.assertEqual(A, self.resolver.lookup('A'))

    def test_make(self):
        """Test making classes."""
        name = 'charlie'
        # Test instantiating with positional dict into kwargs
        self.assertEqual(A(name=name), self.resolver.make('a', {'name': name}))
        # Test instantiating with kwargs
        self.assertEqual(A(name=name), self.resolver.make('a', name=name))

    def test_passthrough(self):
        """Test instances are passed through unmodified."""
        a = A(name='charlie')
        self.assertEqual(a, self.resolver.make(a))

    @unittest.skipIf(tune is None, 'ray[tune] was not installed properly')
    def test_variant_generation(self):
        """Test whether ray tune can generate variants from the search space."""
        search_space = self.resolver.ray_tune_search_space(
            kwargs_search_space=dict(
                name=tune.choice(["charlie", "max"]),
            ),
        )
        for spec in itertools.islice(tune.suggest.variant_generator.generate_variants(search_space), 2):
            config = {
                k[0]: v
                for k, v in spec[0].items()
            }
            query = config.pop("query")
            instance = self.resolver.make(query=query, pos_kwargs=config)
            self.assertIsInstance(instance, Base)
