"""Test utilities."""

import enum
import unittest
from collections import Counter, defaultdict

from class_resolver.utils import (
    add_doc_note_about_resolvers,
    get_subclasses,
    is_private,
    normalize_with_default,
    same_module,
)
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


class DecoratorTests(unittest.TestCase):
    """Decorator tests."""

    @staticmethod
    def f(model, model_kwargs) -> None:
        """Do something, and also use model."""
        pass

    @staticmethod
    def f_no_doc(model, model_kwargs) -> None:  # noqa: D102
        pass

    def test_decorator(self):
        """Test decorator."""
        old_doc = self.f.__doc__
        for params in [["model"], [("model", "model_kwargs")]]:
            decorator = add_doc_note_about_resolvers(*params, resolver_name="model_resolver")
            f_dec = decorator(self.f)
            # note: the decorator modifies the doc string in-place...
            # check that the doc string got extended
            self.assertNotEqual(f_dec.__doc__, old_doc)
            self.assertTrue(f_dec.__doc__.startswith(old_doc))
            # revert for next time
            self.f.__doc__ = old_doc

    def test_error(self):
        """Test error handling."""
        for params in [
            # empty
            [],
            # one-element tuple
            [[("model",)]],
            # three element tuple
            [[("model", "model_kwargs", "model_kwargs2")]],
        ]:
            with self.assertRaises(ValueError):
                add_doc_note_about_resolvers(*params, resolver_name="model_resolver")

    def test_error_decoration(self):
        """Test errors when decorating."""
        # missing docstring
        with self.assertRaises(ValueError):
            add_doc_note_about_resolvers("model", resolver_name="model_resolver")(self.f_no_doc)
        # non-existing parameter name
        with self.assertRaises(ValueError):
            add_doc_note_about_resolvers("interaction", resolver_name="model_resolver")(self.f)
