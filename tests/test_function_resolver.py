# -*- coding: utf-8 -*-

"""Tests for the function resolver."""

import operator
import unittest

from class_resolver import FunctionResolver


def add_one(x: int) -> int:
    """Add one to the number."""
    return x + 1


def add_two(x: int) -> int:
    """Add two to the number."""
    return x + 2


def add_three(x: int) -> int:
    """Add three to the number."""
    return x + 3


def add_y(x: int, y: int) -> int:
    """Add y to the number."""
    return x + y


class TestFunctionResolver(unittest.TestCase):
    """Tests for the function resolver."""

    def setUp(self) -> None:
        """Set up the resolver class."""
        self.resolver = FunctionResolver([add_one, add_two, add_y])

    def test_contents(self):
        """Test the functions."""
        self.assertIn(add_one, set(self.resolver))

    def test_lookup(self):
        """Test looking up functions."""
        self.assertEqual(add_one, self.resolver.lookup("add_one"))
        self.assertEqual(add_one, self.resolver.lookup("ADD_ONE"))
        with self.assertRaises(ValueError):
            self.resolver.lookup(None)
        with self.assertRaises(KeyError):
            self.resolver.lookup("missing")
        with self.assertRaises(TypeError):
            self.resolver.lookup(3)

    def test_default_lookup(self):
        """Test lookup with default."""
        resolver = FunctionResolver([add_one, add_two, add_y], default=add_two)
        self.assertEqual(add_one, resolver.lookup("add_one"))
        self.assertEqual(add_one, resolver.lookup("ADD_ONE"))
        self.assertEqual(add_two, resolver.lookup(None))
        with self.assertRaises(KeyError):
            resolver.lookup("missing")
        with self.assertRaises(TypeError):
            resolver.lookup(3)

    def test_make(self):
        """Test making classes."""
        for x in range(10):
            f1 = self.resolver.make("add_y", {"y": 1})
            self.assertEqual(add_one(x), f1(x))
            # Test instantiating with kwargs
            f2 = self.resolver.make("add_y", y=1)
            self.assertEqual(add_one(x), f2(x))

    def test_make_safe(self):
        """Test the make_safe function, which always returns none on none input."""
        self.assertIsNone(self.resolver.make_safe(None))
        self.assertIsNone(FunctionResolver([add_one, add_two], default=add_two).make_safe(None))

    def test_passthrough(self):
        """Test instances are passed through unmodified."""
        for x in range(10):
            self.assertEqual(add_one(x), self.resolver.make(add_one)(x))

    def test_registration_synonym(self):
        """Test failure of registration."""
        self.resolver.register(add_three, synonyms={"add_trio"})
        for x in range(10):
            self.assertEqual(add_three(x), self.resolver.make("add_trio")(x))

    def test_registration_failure(self):
        """Test failure of registration."""
        with self.assertRaises(KeyError):
            self.resolver.register(add_one)

        def _new_fn(x: int) -> int:
            return x + 1

        with self.assertRaises(KeyError):
            self.resolver.register(_new_fn, synonyms={"add_one"})

    def test_entrypoints(self):
        """Test loading from entrypoints."""
        resolver = FunctionResolver.from_entrypoint("class_resolver_demo")
        self.assertEqual({"add", "sub", "mul"}, set(resolver.lookup_dict))
        self.assertEqual(set(), set(resolver.synonyms))
        self.assertNotIn("expected_failure", resolver.lookup_dict)

    def test_late_entrypoints(self):
        """Test loading late entrypoints."""
        resolver = FunctionResolver([operator.add, operator.sub])
        self.assertEqual({"add", "sub"}, set(resolver.lookup_dict))
        resolver.register_entrypoint("class_resolver_demo")
        self.assertEqual({"add", "sub", "mul"}, set(resolver.lookup_dict))
        self.assertEqual(set(), set(resolver.synonyms))
        self.assertNotIn("expected_failure", resolver.lookup_dict)
