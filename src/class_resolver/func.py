# -*- coding: utf-8 -*-

"""A resolver for functions."""

from functools import partial
from typing import (
    Callable,
    Collection,
    Dict,
    Generic,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
)

from .api import Hint, OptionalKwargs, normalize_string

__all__ = [
    "FunctionResolver",
]

X = TypeVar("X", bound=Callable)


class FunctionResolver(Generic[X]):
    """A resolver for functions."""

    def __init__(
        self,
        functions: Collection[X],
        *,
        default: Optional[X] = None,
        synonyms: Optional[Mapping[str, X]] = None,
    ) -> None:
        """Initialize the resolver.

        :param functions: The functions to registry.
        :param default: The default
        :param synonyms: The optional synonym dictionary
        """
        self.default = default
        self.lookup_dict: Dict[str, X] = {}
        self.synonyms = dict(synonyms or {})
        for func in functions:
            self.register(func)

    def normalize_func(self, func: X) -> str:
        """Normalize a function to a name."""
        return normalize_string(func.__name__)

    def register(
        self, func: X, synonyms: Optional[Iterable[str]] = None, raise_on_conflict: bool = True
    ):
        """Register an additional function with this resolver.

        :param func: The function to register
        :param synonyms: An optional iterable of synonyms to add for the class
        :param raise_on_conflict: Determines the behavior when a conflict is encountered on either
            the normalized class name or a synonym. If true, will raise an exception. If false, will
            simply disregard the entry.

        :raises KeyError: If ``raise_on_conflict`` is true and there's a conflict in either the class
            name or a synonym name.
        """
        key = self.normalize_func(func)
        if key in self.lookup_dict:
            if raise_on_conflict:
                raise KeyError(
                    f"This resolver already contains a class with key {key}: {self.lookup_dict[key]}"
                )
            else:
                return

        self.lookup_dict[key] = func
        for synonym in synonyms or []:
            if synonym in self.synonyms:
                if raise_on_conflict:
                    raise KeyError(
                        f"This resolver already contains synonym {synonym} for {self.synonyms[synonym]}"
                    )
                else:
                    continue
            self.synonyms[synonym] = func

    def lookup(self, query: Hint[X]) -> X:
        """Lookup a function."""
        if query is None:
            if self.default is None:
                raise ValueError("No default is set")
            return self.default
        elif callable(query):
            return query  # type: ignore
        elif isinstance(query, str):
            key = normalize_string(query)
            if key in self.lookup_dict:
                return self.lookup_dict[key]
            elif key in self.synonyms:
                return self.synonyms[key]
            else:
                raise KeyError(
                    f"{query} is an invalid. Try one of: {sorted(self.lookup_dict.keys())}"
                )
        else:
            raise TypeError(f"Invalid function: {type(query)} - {query}")

    def make(self, query: Hint[X], pos_kwargs: OptionalKwargs = None, **kwargs) -> X:
        """Make a function with partial bindings to the given kwargs."""
        if query is None or isinstance(query, str) or callable(query):
            func: X = self.lookup(query)
            if pos_kwargs or kwargs:
                return partial(func, **(pos_kwargs or {}), **kwargs)  # type: ignore
            return func
        return query
