# -*- coding: utf-8 -*-

"""A resolver for functions."""

from functools import partial
from typing import Callable, Optional, TypeVar

from .base import BaseResolver
from .utils import Hint, OptionalKwargs

__all__ = [
    "FunctionResolver",
]

X = TypeVar("X", bound=Callable)


class FunctionResolver(BaseResolver[X]):
    """A resolver for functions."""

    def extract_name(self, element: X) -> str:
        """Get the name for an element."""
        return element.__name__

    def lookup(self, query: Hint[X], default: Optional[X] = None) -> X:
        """Lookup a function."""
        if query is None:
            if default is not None:
                return default
            elif self.default is not None:
                return self.default
            else:
                raise ValueError("No default is set")
        elif callable(query):
            return query  # type: ignore
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

    def make(self, query: Hint[X], pos_kwargs: OptionalKwargs = None, **kwargs) -> X:
        """Make a function with partial bindings to the given kwargs."""
        func: X = self.lookup(query)
        if pos_kwargs or kwargs:
            return partial(func, **(pos_kwargs or {}), **kwargs)  # type: ignore
        return func

    def make_safe(self, query: Hint[X], pos_kwargs: OptionalKwargs = None, **kwargs) -> Optional[X]:
        """Run make, but pass through a none query."""
        if query is None:
            return None
        return self.make(query=query, pos_kwargs=pos_kwargs, **kwargs)
