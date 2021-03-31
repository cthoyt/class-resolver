# -*- coding: utf-8 -*-

"""Tests for the class resolver."""

import unittest

import ray
import ray.tune

from class_resolver import Resolver


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

    def test_ray(self):
        """Test case for the ray.tune search space."""
        ray.init(local_mode=True)
        analysis = ray.tune.run(
            ray.tune.with_parameters(self._dummy_training_function, resolver=self.resolver),
            config=dict(
                choice=self.resolver.ray_tune_search_space(
                    kwargs_search_space=dict(
                        name=ray.tune.choice(["charlie", "max"]),
                    ),
                ),
            ),
            ray_auto_init=False,
        )

        analysis.get_best_config(metric="mean_loss", mode="min")

    @staticmethod
    def _dummy_training_function(config, resolver):
        """A dummy training function without actual training."""
        # instantiate from configuration
        to_resolve = config["choice"]
        if isinstance(to_resolve, dict):
            query = to_resolve.pop("query")
            kwargs = to_resolve
        else:
            query = to_resolve
            kwargs = None
        instance = resolver.make(query, pos_kwargs=kwargs)
        if instance.name == "charlie":
            mean_loss = 1.0
        else:
            mean_loss = 2.0
        ray.tune.report(mean_loss=mean_loss)
