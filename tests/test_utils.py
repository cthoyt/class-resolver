# -*- coding: utf-8 -*-

"""Test utilities."""

import enum
import unittest
from collections import Counter, defaultdict

from class_resolver.utils import get_subclasses, same_module
from tests._private_extras import PrivateDict


class TestUtilities(unittest.TestCase):
    """Test utilities."""

    def test_same_module(self):
        """Test getting subclasses."""
        self.assertFalse(same_module(Counter, dict))
        self.assertTrue(same_module(Counter, defaultdict))

    def test_get_subclasses(self):
        """Test getting subclasses."""
        self.assertTrue(issubclass(PrivateDict, dict))

        self.assertNotIn(Counter, set(get_subclasses(dict, exclude_external=True)))
        self.assertIn(Counter, set(get_subclasses(dict, exclude_external=False)))

        self.assertIn(
            PrivateDict, set(get_subclasses(dict, exclude_external=False, exclude_private=False))
        )
        self.assertNotIn(
            PrivateDict, set(get_subclasses(dict, exclude_external=False, exclude_private=True))
        )
        self.assertNotIn(
            PrivateDict, set(get_subclasses(dict, exclude_external=True, exclude_private=False))
        )
        self.assertNotIn(
            PrivateDict, set(get_subclasses(dict, exclude_external=True, exclude_private=True))
        )

        self.assertIn(
            enum._EnumDict, set(get_subclasses(dict, exclude_external=False, exclude_private=False))
        )
        self.assertNotIn(
            enum._EnumDict, set(get_subclasses(dict, exclude_external=False, exclude_private=True))
        )
        self.assertNotIn(
            enum._EnumDict, set(get_subclasses(dict, exclude_external=True, exclude_private=False))
        )
        self.assertNotIn(
            enum._EnumDict, set(get_subclasses(dict, exclude_external=True, exclude_private=True))
        )
