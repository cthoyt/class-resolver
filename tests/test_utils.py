"""Test utilities."""

from __future__ import annotations

import enum
import sys
import unittest
from collections import Counter, defaultdict
from collections.abc import Sequence

from class_resolver.utils import get_subclasses, is_private, normalize_with_default, same_module
from tests._private_extras import PrivateDict


class TestUtilities(unittest.TestCase):
    """Test utilities."""

    def test_is_private(self) -> None:
        """Test whether a module should be considered private."""
        self.assertFalse(is_private("A", "", main_is_private=False))
        self.assertFalse(is_private("A", "", main_is_private=True))
        self.assertTrue(is_private("_module", "", main_is_private=True))
        self.assertTrue(is_private("_module", "", main_is_private=True))
        self.assertTrue(is_private("A", "B._module", main_is_private=True))
        self.assertTrue(is_private("A", "__main__", main_is_private=True))
        self.assertFalse(is_private("A", "__main__", main_is_private=False))
        self.assertTrue(is_private("_A", "__main__", main_is_private=True))
        self.assertTrue(is_private("_A", "__main__", main_is_private=False))

    def test_same_module(self) -> None:
        """Test getting subclasses."""
        self.assertFalse(same_module(Counter, dict))
        self.assertTrue(same_module(Counter, defaultdict))

    def test_get_subclasses(self) -> None:
        """Test getting subclasses."""
        self.assertTrue(issubclass(PrivateDict, dict))

        self.assertNotIn(Counter, set(get_subclasses(dict, exclude_external=True)))
        self.assertIn(Counter, set(get_subclasses(dict, exclude_external=False)))

        self.assertIn(PrivateDict, set(get_subclasses(dict, exclude_external=False, exclude_private=False)))
        self.assertNotIn(PrivateDict, set(get_subclasses(dict, exclude_external=False, exclude_private=True)))
        self.assertNotIn(PrivateDict, set(get_subclasses(dict, exclude_external=True, exclude_private=False)))
        self.assertNotIn(PrivateDict, set(get_subclasses(dict, exclude_external=True, exclude_private=True)))

        if sys.version_info < (3, 13):
            # in Python 3.13, this class gets renamed and the test doesn't work anymore
            self.assertIn(enum._EnumDict, set(get_subclasses(dict, exclude_external=False, exclude_private=False)))
            self.assertNotIn(enum._EnumDict, set(get_subclasses(dict, exclude_external=False, exclude_private=True)))
            self.assertNotIn(enum._EnumDict, set(get_subclasses(dict, exclude_external=True, exclude_private=False)))
            self.assertNotIn(enum._EnumDict, set(get_subclasses(dict, exclude_external=True, exclude_private=True)))

    def test_normalize_with_defaults(self) -> None:
        """Tests for normalize with defaults."""
        # choice and default are None -> error
        with self.assertRaises(ValueError):
            normalize_with_default(choice=None, default=None)

        # choice is None -> use default *and* default_kwargs irrespective of kwargs
        default_kwargs = {"b": 3}
        for choice_kwargs in (None, {"a": 5}):
            cls, kwargs = normalize_with_default(
                choice=None, kwargs=choice_kwargs, default=Counter, default_kwargs=default_kwargs
            )
            self.assertIs(cls, Counter)
            self.assertIs(kwargs, default_kwargs)

    def test_normalize_with_defaults_2(self) -> None:
        """Tests for normalize with defaults."""
        default_kwargs_tests: Sequence[None | dict[str, int]] = [None, {"b": 3}]
        # choice is not None -> return choice and kwargs
        choice_kwargs = {"a": 5}
        for default_kwargs in default_kwargs_tests:
            cls, kwargs = normalize_with_default(
                choice=dict, kwargs=choice_kwargs, default=Counter, default_kwargs=default_kwargs
            )
            self.assertIs(cls, dict)
            self.assertIs(kwargs, choice_kwargs)
