# -*- coding: utf-8 -*-

"""Test utilities."""

import enum
import unittest
from collections import Counter, defaultdict

from class_resolver.utils import get_subclasses, normalize_with_default, same_module
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

    def test_normalize_with_defaults(self):
        """Tests for normalize with defaults."""
        # choice and default are None -> error
        with self.assertRaises(ValueError):
            normalize_with_default(choice=None, default=None)

        # choice is None -> use default *and* default_kwargs irrespective of kwargs
        default_kwargs = dict(b=3)
        for choice_kwargs in (None, dict(a=5)):
            cls, kwargs = normalize_with_default(
                choice=None, kwargs=choice_kwargs, default=Counter, default_kwargs=default_kwargs
            )
            self.assertIs(cls, Counter)
            self.assertIs(kwargs, default_kwargs)

        # choice is not None -> return choice and kwargs
        choice_kwargs = dict(a=5)
        for default_kwargs in (None, dict(b=3)):
            cls, kwargs = normalize_with_default(
                choice=dict, kwargs=choice_kwargs, default=Counter, default_kwargs=default_kwargs
            )
            self.assertIs(cls, dict)
            self.assertIs(kwargs, choice_kwargs)
