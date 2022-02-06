# -*- coding: utf-8 -*-

"""A base resolver."""

from abc import ABC, abstractmethod
from typing import Collection, Dict, Generic, Iterable, Iterator, Mapping, Optional, Set

from .utils import Hint, X, make_callback, normalize_string

__all__ = [
    "BaseResolver",
]


class BaseResolver(ABC, Generic[X]):
    """A resolver for arbitrary elements."""

    default: Optional[X]
    #: The mapping from synonyms to the classes indexed by this resolver
    synonyms: Dict[str, X]
    #: The mapping from normalized class names to the classes indexed by this resolver
    lookup_dict: Dict[str, X]
    #: The shared suffix fo all classes derived from the base class
    suffix: Optional[str]

    def __init__(
        self,
        elements: Optional[Iterable[X]] = None,
        *,
        default: Optional[X] = None,
        synonyms: Optional[Mapping[str, X]] = None,
        suffix: Optional[str] = None,
    ):
        """Initialize the resolver.

        :param elements: The elements to register.
        :param default: The optional default element
        :param synonyms: The optional synonym dictionary
        :param suffix: The optional shared suffix of all classes. If None, use the base class' name for it.
            To disable this behaviour, explicitly provide `suffix=""`.
        """
        self.default = default
        self.synonyms = dict(synonyms or {})
        self.lookup_dict = {}
        self.suffix = suffix
        if elements is not None:
            for element in elements:
                self.register(element)

    def __iter__(self) -> Iterator[X]:
        """Iterate over the registered elements."""
        return iter(self.lookup_dict.values())

    @property
    def options(self) -> Set[str]:
        """Return the normalized option names."""
        return set(self.lookup_dict.keys()).union(self.synonyms.keys())

    @abstractmethod
    def extract_name(self, element: X) -> str:
        """Get the name for an element."""

    def extract_synonyms(self, element: X) -> Collection[str]:
        """Get synonyms from an element."""
        return []

    def normalize(self, s: str) -> str:
        """Normalize the string with this resolve's suffix."""
        return normalize_string(s, suffix=self.suffix)

    def register(
        self,
        element: X,
        synonyms: Optional[Iterable[str]] = None,
        raise_on_conflict: bool = True,
    ):
        """Register an additional element with this resolver.

        :param element: The element to register
        :param synonyms: An optional iterable of synonyms to add for the element
        :param raise_on_conflict: Determines the behavior when a conflict is encountered on either
            the normalized element name or a synonym. If true, will raise an exception. If false, will
            simply disregard the entry.

        :raises KeyError: If ``raise_on_conflict`` is true and there's a conflict in either the element
            name or synonym.
        :raises ValueError: If any given synonyms are empty strings
        """
        key = self.normalize(self.extract_name(element))
        if key not in self.lookup_dict:
            self.lookup_dict[key] = element
        elif raise_on_conflict:
            raise KeyError(
                f"This resolver already contains an element with key {key}: {self.lookup_dict[key]}"
            )

        _synonyms = set(synonyms or [])
        _synonyms.update(self.extract_synonyms(element))

        for synonym in _synonyms:
            synonym_key = self.normalize(synonym)
            if not synonym_key:
                raise ValueError(f"Tried to use empty synonym for {element}")
            if synonym_key not in self.synonyms and synonym_key not in self.lookup_dict:
                self.synonyms[synonym_key] = element
            elif raise_on_conflict:
                raise KeyError(
                    f"This resolver already contains synonym {synonym} for {self.synonyms[synonym]}"
                )

    @abstractmethod
    def lookup(self, query: Hint[X], default: Optional[X] = None) -> X:
        """Lookup an element."""

    def get_option(
        self,
        *flags: str,
        default: Hint[X] = None,
        as_string: bool = False,
        **kwargs,
    ):
        """Get a click option for this resolver."""
        if default is None:
            if self.default is None:
                raise ValueError("no default given either from resolver or explicitly")
            default = self.default
        else:
            default = self.lookup(default)
        default = self.extract_name(default)

        import click

        return click.option(
            *flags,
            type=click.Choice(list(self.lookup_dict), case_sensitive=False),
            default=[default] if kwargs.get("multiple") else default,
            show_default=True,
            callback=None if as_string else make_callback(self.lookup),
            **kwargs,
        )
