# -*- coding: utf-8 -*-

"""Tests for the function resolver."""

import unittest

from class_resolver import FunctionResolver


def add_one(x: int) -> int:
    """Add one to the number."""
    return x + 1


def add_two(x: int) -> int:
    """Add two to the number."""
    return x + 2


def add_y(x: int, y: int) -> int:
    """Add y to the number."""
    return x + y


class TestFunctionResolver(unittest.TestCase):
    """Tests for the function resolver."""

    def setUp(self) -> None:
        """Set up the resolver class."""
        self.resolver = FunctionResolver([add_one, add_two, add_y])

    def test_lookup(self):
        """Test looking up functions."""
        self.assertEqual(add_one, self.resolver.lookup("add_one"))
        self.assertEqual(add_one, self.resolver.lookup("ADD_ONE"))

    def test_make(self):
        """Test making classes."""
        for x in range(10):
            f1 = self.resolver.make("add_y", {"y": 1})
            self.assertEqual(add_one(x), f1(x))
            # Test instantiating with kwargs
            f2 = self.resolver.make("add_y", y=1)
            self.assertEqual(add_one(x), f2(x))

    def test_passthrough(self):
        """Test instances are passed through unmodified."""
        for x in range(10):
            self.assertEqual(add_one(x), self.resolver.make(add_one)(x))

    def test_registration_failure(self):
        """Test failure of registration."""
        with self.assertRaises(KeyError):
            self.resolver.register(add_one)

        def _add_one(x: int) -> int:
            return x + 1

        with self.assertRaises(KeyError):
            self.resolver.register(_add_one, synonyms={"add_one"})
