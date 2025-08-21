"""A resolver for functions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, Generic, TypeVar

from typing_extensions import ParamSpec

from .base import BaseResolver
from .utils import Hint, OptionalKwargs

__all__ = [
    "FunctionResolver",
]

P = ParamSpec("P")
T = TypeVar("T")


class FunctionResolver(Generic[P, T], BaseResolver[Callable[P, T], Callable[P, T]]):
    """A resolver for functions."""

    def extract_name(self, element: Callable[P, T]) -> str:
        """Get the name for an element."""
        return element.__name__

    def lookup(self, query: Hint[Callable[P, T]], default: Callable[P, T] | None = None) -> Callable[P, T]:
        """Lookup a function."""
        if query is None:
            return self._default(default)
        elif callable(query):
            return query
        elif isinstance(query, str):
            key = self.normalize(query)
            if key in self.lookup_dict:
                return self.lookup_dict[key]
            elif key in self.synonyms:
                return self.synonyms[key]
            else:
                valid_choices = sorted(self.options)
                raise KeyError(f"{query} is an invalid. Try one of: {valid_choices}")
        else:
            raise TypeError(f"Invalid function: {type(query)} - {query}")

    def make(self, query: Hint[Callable[P, T]], pos_kwargs: OptionalKwargs = None, **kwargs: Any) -> Callable[P, T]:
        """Make a function with partial bindings to the given kwargs."""
        func = self.lookup(query)
        if pos_kwargs or kwargs:
            return partial(func, **(pos_kwargs or {}), **kwargs)
        return func
