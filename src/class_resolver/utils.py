# -*- coding: utf-8 -*-

"""Utilities for the resolver."""

import collections.abc
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

if TYPE_CHECKING:
    import click

__all__ = [
    # Type Hints
    "X",
    "Hint",
    "HintOrType",
    "HintType",
    "InstOrType",
    "Lookup",
    "LookupOrType",
    "LookupType",
    "OptionalKwargs",
    # Functions
    "get_subclasses",
    "normalize_string",
    "upgrade_to_sequence",
    "make_callback",
]

X = TypeVar("X")
Y = TypeVar("Y")

#: A type annotation for either an instance of X or a class of X
InstOrType = Union[X, Type[X]]
#: A type annotation for either an instance of X or name a class X
Lookup = Union[str, X]

LookupType = Lookup[Type[X]]
LookupOrType = Lookup[InstOrType[X]]
Hint = Optional[Lookup[X]]
HintType = Hint[Type[X]]
HintOrType = Hint[InstOrType[X]]
OptionalKwargs = Optional[Mapping[str, Any]]
OneOrSequence = Union[X, Sequence[X]]


def get_subclasses(cls: Type[X], exclude_private: bool = True) -> Iterable[Type[X]]:
    """Get all subclasses.

    :param cls: The ancestor class
    :param exclude_private: If true, will skip any class that comes from a module
        starting with an underscore (i.e., a private module). This is typically
        done when having shadow duplicate classes implemented in C
    :yields: Descendant classes of the ancestor class
    """
    for subclass in cls.__subclasses__():
        yield from get_subclasses(subclass)
        if exclude_private and any(part.startswith("_") for part in subclass.__module__.split(".")):
            continue
        yield subclass


def normalize_string(s: str, *, suffix: Optional[str] = None) -> str:
    """Normalize a string for lookup."""
    s = s.lower().replace("-", "").replace("_", "").replace(" ", "")
    if suffix is not None and s.endswith(suffix.lower()):
        return s[: -len(suffix)]
    return s.strip()


def upgrade_to_sequence(x: Union[X, Sequence[X]]) -> Sequence[X]:
    """Ensure that the input is a sequence.

    :param x: A literal or sequence of literals (don't consider a string x as a sequence)
    :return: If a literal was given, a one element tuple with it in it. Otherwise, return the given value.

    >>> upgrade_to_sequence(1)
    (1,)
    >>> upgrade_to_sequence((1, 2, 3))
    (1, 2, 3)
    >>> upgrade_to_sequence("test")
    ('test',)
    >>> upgrade_to_sequence(tuple("test"))
    ('t', 'e', 's', 't')
    """
    if isinstance(x, str):
        return (x,)  # type: ignore
    elif isinstance(x, collections.abc.Sequence):
        return x
    else:
        return (x,)


def make_callback(f: Callable[[X], Y]) -> Callable[["click.Context", "click.Parameter", X], Y]:
    """Make a click-appropriate callback."""

    def _callback(_ctx: "click.Context", _param: "click.Parameter", value: X) -> Y:
        return f(value)

    return _callback
