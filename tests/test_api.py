# -*- coding: utf-8 -*-


"""Tests for the class resolver."""

import unittest
from dataclasses import dataclass

from class_resolver import Resolver


class Base:
    """A letter class."""


@dataclass
class A(Base):
    """A base class."""
    name: str


@dataclass
class B(Base):
    """B base class."""
    name: str


@dataclass
class C(Base):
    """C base class."""
    name: str


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
